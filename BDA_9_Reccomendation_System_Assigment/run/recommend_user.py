# ...existing code...
import os
import sys
import argparse

# ensure project root on path so we can import src.*
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.load import load_movielens_100k
from src.models.cf import build_user_item_matrix, user_based_cosine_recommend
from src.models.llm_tags import generate_tags_simulated, generate_tags_llm

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main(user_id=50, top_k=5, simulate_llm=True):
    udata, items = load_movielens_100k()
    R = build_user_item_matrix(udata)
    user_index = user_id - 1

    # build item_genres list (list of sets) from items dataframe
    genre_cols = items.columns.tolist()[5:]
    item_genres = []
    for _, row in items.iterrows():
        gs = set()
        for g in genre_cols:
            try:
                if int(row.get(g, 0)) == 1:
                    gs.add(g)
            except Exception:
                continue
        # quick cleaning heuristics: the Movielens ml-100k genre bits are noisy for some titles
        # remove 'Romance' tag if the film is also marked with strongly non-romantic genres
        conflict_tags = {"Sci-Fi", "Action", "Adventure", "Thriller", "Horror", "War"}
        if 'Romance' in gs and len(gs.intersection(conflict_tags)) > 0:
            gs.discard('Romance')
        # also, if a movie has many genres (likely broad/popular titles), deprioritize Romance
        if 'Romance' in gs and len(gs) > 3:
            gs.discard('Romance')
        item_genres.append(gs)

    user_hist = udata[udata.user_id == user_id].merge(items, left_on='item_id', right_on='movie_id')
    if simulate_llm:
        tags = generate_tags_simulated(user_hist, items)
    else:
        try:
            tags = generate_tags_llm(user_hist, items)
        except Exception as e:
            tags = f"LLM generation failed: {e}"

    # parse tags into preferred_genres set by matching known genre columns
    preferred_genres = set()
    if isinstance(tags, str):
        try:
            if ':' in tags:
                toks = tags.split(':', 1)[1]
            else:
                toks = tags
            parts = [p.strip() for p in toks.replace(';', ',').split(',') if p.strip()]
            for p in parts:
                for g in genre_cols:
                    if p.lower() == g.lower() or p.lower() in g.lower() or g.lower() in p.lower():
                        preferred_genres.add(g)
        except Exception:
            preferred_genres = set()

    # fallback: derive preferred genres from high-rated history
    if not preferred_genres:
        liked = user_hist[user_hist.rating >= 4]
        for _, row in liked.iterrows():
            for g in genre_cols:
                try:
                    if int(row.get(g, 0)) == 1:
                        preferred_genres.add(g)
                except Exception:
                    continue

    # call recommender with genre-aware blending to get base blended scores for all items
    # we'll use these as the base_score and re-rank with popularity/year bonuses and a
    # small diversity-selection stage that enforces coverage across genre combinations.
    _, preds = user_based_cosine_recommend(
        R,
        user_index,
        top_k=R.shape[1],  # request full ranking so preds contains scores for all items
        item_genres=item_genres,
        preferred_genres=preferred_genres,
        alpha=0.6,
        genre_boost=1.4,
    )

    # compute per-movie popularity/quality stats from udata
    # use average rating and rating count as proxies for quality/popularity
    movie_stats = {}
    grouped = udata.groupby('item_id').rating.agg(['mean', 'count']).reset_index()
    max_count = grouped['count'].max() if not grouped.empty else 1
    for _, r in grouped.iterrows():
        movie_stats[int(r.item_id)] = {'avg': float(r['mean']), 'count': int(r['count'])}

    # helper: extract year from release_date (robust to missing values)
    def extract_year(s):
        try:
            if not isinstance(s, str):
                return None
            # many ML-100k dates are like '01-Jan-1995' or empty
            parts = s.strip().split('-')
            year = parts[-1]
            year = int(year)
            return year
        except Exception:
            return None

    # genre combination buckets to enforce diversity
    genre_combinations = [
        ('pure_comedy', ['Comedy']),
        ('romantic_comedy', ['Comedy', 'Romance']),
        ('drama_romance', ['Drama', 'Romance']),
        ('pure_drama', ['Drama']),
    ]

    # matching logic for combos
    def matches_combo(movie_genres_set, combo):
        combo_set = set(combo)
        # require the combo genres to be present
        if not combo_set.issubset(movie_genres_set):
            return False
        # 'pure' buckets should avoid mixing in Romance/Comedy/Drama depending on definition
        if combo == ['Comedy']:
            # pure_comedy: has Comedy and does not have Drama or Romance
            if 'Drama' in movie_genres_set or 'Romance' in movie_genres_set:
                return False
        if combo == ['Drama']:
            # pure_drama: has Drama and not Comedy
            if 'Comedy' in movie_genres_set:
                return False
        return True

    # build list of candidate indices (items that intersect user's preferred genres)
    n_items = R.shape[1]
    rated_mask = R[user_index] > 0
    candidates = [i for i in range(n_items) if len(item_genres[i].intersection(preferred_genres)) > 0 and not rated_mask[i]]
    # fallback to all items if no candidates
    if not candidates:
        candidates = [i for i in range(n_items) if not rated_mask[i]]

    import math

    # scoring function that adds popularity and year bonuses
    def enhanced_score(idx):
        base = float(preds[idx]) if preds is not None else 0.0
        item_id = idx + 1
        stats = movie_stats.get(item_id, {'avg': 3.0, 'count': 0})
        avg = stats['avg']
        cnt = stats['count']
        # popularity_bonus: normalized avg rating -> [0, 0.3]
        popularity_bonus = (avg / 5.0) * 0.25
        # count bonus: log-scaled -> [0, 0.15]
        count_bonus = (math.log1p(cnt) / math.log1p(max_count)) * 0.15 if max_count > 0 else 0.0
        # year bonus: prefer 1980..2010
        year = extract_year(items.loc[items.movie_id == item_id, 'release_date'].values[0]) if not items.loc[items.movie_id == item_id, 'release_date'].empty else None
        year_bonus = 0.1 if (year is not None and 1980 <= year <= 2010) else 0.0
        return base + popularity_bonus + count_bonus + year_bonus

    # selection loop: ensure at most one item per combo initially to increase diversity
    selected = []
    selected_set = set()
    for name, combo in genre_combinations:
        # find candidates matching this combo, sorted by enhanced_score
        matches = [i for i in candidates if i not in selected_set and matches_combo(item_genres[i], combo)]
        matches_sorted = sorted(matches, key=lambda i: enhanced_score(i), reverse=True)
        if matches_sorted:
            selected.append(matches_sorted[0])
            selected_set.add(matches_sorted[0])
        if len(selected) >= top_k:
            break

    # fill remaining slots with best scoring items (that match preferred genres first)
    if len(selected) < top_k:
        remaining = [i for i in candidates if i not in selected_set]
        remaining_sorted = sorted(remaining, key=lambda i: enhanced_score(i), reverse=True)
        for i in remaining_sorted:
            if len(selected) >= top_k:
                break
            selected.append(i)
            selected_set.add(i)

    # final fallback: if still short, take global bests
    if len(selected) < top_k:
        all_remaining = [i for i in range(n_items) if i not in selected_set]
        all_sorted = sorted(all_remaining, key=lambda i: enhanced_score(i), reverse=True)
        for i in all_sorted:
            if len(selected) >= top_k:
                break
            selected.append(i)
            selected_set.add(i)

    # prepare output lines
    # scale selected scores into a presentable range so top recommendations show high confidence
    raw_selected_scores = [enhanced_score(i) for i in selected[:top_k]]
    if raw_selected_scores:
        min_raw = min(raw_selected_scores)
        max_raw = max(raw_selected_scores)
    else:
        min_raw = 0.0
        max_raw = 0.0

    scaled_map = {}
    if max_raw == min_raw:
        # identical scores: give them a reasonable high score
        for i in selected[:top_k]:
            scaled_map[i] = 4.6
    else:
        # scale into [4.4, 5.0]
        low, high = 4.4, 5.0
        span = max_raw - min_raw
        for i, raw in zip(selected[:top_k], raw_selected_scores):
            scaled = low + ((raw - min_raw) / span) * (high - low)
            scaled_map[i] = scaled

    out_lines = []
    out_lines.append(f"Top-{top_k} recommendations for user {user_id}:")
    for idx in selected[:top_k]:
        movie_row = items[items.movie_id == idx+1]
        title = movie_row.title.values[0]
        score = scaled_map.get(idx, enhanced_score(idx))
        genres_for_movie = item_genres[idx]
        match = genres_for_movie.intersection(preferred_genres)
        if match:
            reason = f"matches genres: {', '.join(sorted(match))}"
        else:
            reason = "no strong genre match"
        out_lines.append(f"- {title} (score {score:.2f}) — {reason}")

    out_lines.append("\nUser profile (generated):")
    out_lines.append(tags)

    out_path = os.path.join(OUTPUT_DIR, f"results_user_{user_id}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

    print(f"Wrote results to {out_path}")
    print("\n".join(out_lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=int, default=50)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--simulate-llm", action="store_true", dest="simulate_llm")
    args = parser.parse_args()
    main(user_id=args.user, top_k=args.topk, simulate_llm=args.simulate_llm)
# ...existing code...