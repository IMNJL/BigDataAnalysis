import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Set, Optional


def build_user_item_matrix(udata, n_users=943, n_items=1682):
    R = np.zeros((n_users, n_items))
    for _, row in udata.iterrows():
        R[int(row.user_id) - 1, int(row.item_id) - 1] = row.rating
    return R


def _genre_match_score(item_genres: Set[str], preferred_genres: Set[str]) -> float:
    """Return a normalized genre match score between 0 and 1."""
    if not preferred_genres or not item_genres:
        return 0.0
    inter = item_genres.intersection(preferred_genres)
    # score = fraction of preferred genres present in item (could also use other heuristics)
    return len(inter) / len(preferred_genres)


def user_based_cosine_recommend(
    R: np.ndarray,
    user_index: int,
    top_k: int = 5,
    item_genres: Optional[List[Set[str]]] = None,
    preferred_genres: Optional[Set[str]] = None,
    alpha: float = 0.75,
    genre_boost: float = 1.2,
):
    """
    User-based CF with optional genre-aware re-ranking.

    Parameters:
    - R: user-item rating matrix (users x items)
    - user_index: index of target user (0-based)
    - top_k: number of recommendations to return
    - item_genres: optional list (length = n_items) of sets of genre strings for each item
    - preferred_genres: optional set of user's preferred genres (strings)
    - alpha: weight for CF score when blending with genre score (0..1). final = alpha*cf + (1-alpha)*genre
    - genre_boost: multiplicative boost applied to items that have any genre match (optional)

    Returns (top_idx, preds): indices of recommended items and the blended score array.
    """
    # compute cosine similarities between users (rows)
    sim = cosine_similarity(R)
    sim[user_index, user_index] = 0.0

    # compute weighted CF prediction (this is on the original rating scale)
    denom = sim[user_index].sum()
    if denom == 0:
        # fallback: predict user's mean rating if we have any ratings, otherwise global mean 3.0
        user_ratings = R[user_index]
        if np.count_nonzero(user_ratings) > 0:
            user_mean = user_ratings[user_ratings > 0].mean()
        else:
            user_mean = 3.0
        cf_preds = np.full(R.shape[1], user_mean)
    else:
        cf_preds = sim[user_index].dot(R) / denom

    # mask already rated items for ranking, but keep cf_preds values for reporting
    rated = R[user_index] > 0

    # compute genre-based score (0..1), then scale it to the rating scale (assume max_rating=5)
    n_items = R.shape[1]
    genre_score = np.zeros(n_items)
    if item_genres is not None and preferred_genres is not None and len(preferred_genres) > 0:
        for i in range(n_items):
            gscore = _genre_match_score(item_genres[i], preferred_genres)
            if gscore > 0:
                gscore *= genre_boost
            genre_score[i] = gscore
        # genre_score is already in 0..(possibly >1 if boosted); clamp to [0,1]
        genre_score = np.clip(genre_score, 0.0, 1.0)
    else:
        genre_score = np.zeros(n_items)

    # scale genre score to rating range
    max_rating = 5.0
    genre_score_scaled = genre_score * max_rating

    # blended score on rating scale
    blended = alpha * cf_preds + (1.0 - alpha) * genre_score_scaled

    # deprioritize already rated items for selection
    blended_for_ranking = blended.copy()
    blended_for_ranking[rated] = -np.inf

    top_idx = np.argsort(blended_for_ranking)[-top_k:][::-1]
    return top_idx, blended
