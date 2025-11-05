import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def build_user_item_matrix(udata, n_users=943, n_items=1682):
    R = np.zeros((n_users, n_items))
    for _, row in udata.iterrows():
        R[int(row.user_id)-1, int(row.item_id)-1] = row.rating
    return R

def user_based_cosine_recommend(R, user_index, top_k=5):
    # compute cosine similarities
    sim = cosine_similarity(R)
    sim[user_index, user_index] = 0.0
    denom = sim[user_index].sum()
    if denom == 0:
        preds = np.zeros(R.shape[1])
    else:
        preds = sim[user_index].dot(R) / denom
    rated = R[user_index] > 0
    preds[rated] = -np.inf
    top_idx = np.argsort(preds)[-top_k:][::-1]
    return top_idx, preds
