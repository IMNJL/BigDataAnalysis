from collections import Counter

# Simulated tag generator (safe, fast). Use real LLM call if needed.
def generate_tags_simulated(user_hist, items, top_n=3):
    liked = user_hist[user_hist.rating >= 4]
    if liked.empty:
        liked = user_hist.sort_values(by='rating', ascending=False).head(10)
    # genres are columns in items starting at index 5
    genre_cols = items.columns.tolist()[5:]
    genre_counts = Counter()
    for _, row in liked.iterrows():
        for g in genre_cols:
            try:
                if int(row.get(g, 0)) == 1:
                    genre_counts[g] += 1
            except Exception:
                continue
    most = [g for g, _ in genre_counts.most_common(top_n)]
    if not most:
        return "No strong genre preference detected."
    return "Prefers: " + ", ".join(most)

# Placeholder for real LLM invocation (not executed by default)
def generate_tags_llm(user_hist, items, model_name="google/flan-t5-small", device=-1):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    lines = []
    for _, row in user_hist.head(20).iterrows():
        lines.append(f"{row.title} (rating {int(row.rating)})")
    prompt = "Given the following user movie history, produce a short summary of preferences (genres, themes) in 1-2 phrases.\n\n" \
             "Movies:\n" + "\n".join(lines) + "\n\nSummary:"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    out = gen(prompt, max_length=64, do_sample=False)
    return out[0]['generated_text']
