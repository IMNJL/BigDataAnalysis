# inspector.py
import sqlite3

def print_examples(db_file="hn_cleaned.db", n=10):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM hn_items")
    total = cur.fetchone()[0]
    print(f"=== Total {total} items in {db_file} ===")
    cur.execute("SELECT id, title, substr(cleaned_text,1,120), url FROM hn_items ORDER BY fetched_at DESC LIMIT ?", (n,))
    rows = cur.fetchall()
    for r in rows:
        print(r)
    conn.close()
