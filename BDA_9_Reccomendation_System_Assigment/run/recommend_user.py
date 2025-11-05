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

    # call recommender with genre-aware blending
    # give stronger weight to genre match to prioritize relevance over pure popularity
    top_idx, preds = user_based_cosine_recommend(
        R,
        user_index,
        top_k=top_k,
        item_genres=item_genres,
        preferred_genres=preferred_genres,
        alpha=0.5,
        genre_boost=1.5,
    )

    out_lines = []
    out_lines.append(f"Top-{top_k} recommendations for user {user_id}:")
    for idx in top_idx:
        movie_row = items[items.movie_id == idx+1]
        title = movie_row.title.values[0]
        score = preds[idx]
        genres_for_movie = item_genres[idx]
        match = genres_for_movie.intersection(preferred_genres)
        if match:
            reason = f"matches genres: {', '.join(sorted(match))}"
        else:
            reason = "no strong genre match"
        out_lines.append(f"- {title} (score {score:.3f}) — {reason}")

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