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
    top_idx, preds = user_based_cosine_recommend(R, user_index, top_k=top_k)

    out_lines = []
    out_lines.append(f"Top-{top_k} recommendations for user {user_id}:")
    for idx in top_idx:
        movie_row = items[items.movie_id == idx+1]
        title = movie_row.title.values[0]
        score = preds[idx]
        out_lines.append(f"- {title} (predicted score {score:.3f})")

    user_hist = udata[udata.user_id == user_id].merge(items, left_on='item_id', right_on='movie_id')
    if simulate_llm:
        tags = generate_tags_simulated(user_hist, items)
    else:
        try:
            tags = generate_tags_llm(user_hist, items)
        except Exception as e:
            tags = f"LLM generation failed: {e}"

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