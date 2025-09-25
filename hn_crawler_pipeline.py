# hn_crawler_pipeline.py
"""
Pipeline modes:
- top: crawl top stories (обычно до ~500 id)
- archive: crawl последние N записей начиная от maxitem.json

1) Crawl -> save raw JSON into hn_raw.db
2) Parse & Clean -> read hn_raw.db, clean text -> save into hn_cleaned.db
3) Print examples
"""

import asyncio
import aiohttp
import aiosqlite
import time
import html
import json
import re
from bs4 import BeautifulSoup
from typing import Optional

BASE = "https://hacker-news.firebaseio.com/v0"
TOP_STORIES = f"{BASE}/topstories.json"
MAXITEM = f"{BASE}/maxitem.json"
ITEM_URL = f"{BASE}/item/{{id}}.json"

# File names
RAW_DB = "hn_raw.db"
CLEAN_DB = "hn_cleaned.db"

# Config
CONCURRENT_REQUESTS = 64
REQUEST_DELAY = 0.1     # politeness delay
DEFAULT_TOP_LIMIT = 500
DEFAULT_ARCHIVE_LIMIT = 100000
SAMPLE_PRINT = 10

# ---------- Utilities ----------
def clean_html_fragment(raw_html: Optional[str]) -> str:
    if not raw_html:
        return ""
    un = html.unescape(raw_html)
    soup = BeautifulSoup(un, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", "[email_removed]", text)
    text = re.sub(r"[\$€£¥]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# ---------- DB schemas ----------
RAW_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS raw_items (
    id INTEGER PRIMARY KEY,
    raw_json TEXT,
    fetched_at REAL
);
"""
CLEAN_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS hn_items (
    id INTEGER PRIMARY KEY,
    type TEXT,
    by TEXT,
    time INTEGER,
    title TEXT,
    raw_text TEXT,
    cleaned_text TEXT,
    url TEXT,
    fetched_at REAL
);
"""
INSERT_RAW_SQL = "INSERT OR REPLACE INTO raw_items (id, raw_json, fetched_at) VALUES (?, ?, ?)"
INSERT_CLEAN_SQL = """
INSERT OR REPLACE INTO hn_items (id, type, by, time, title, raw_text, cleaned_text, url, fetched_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# ---------- Fetch ----------
async def fetch_json(session: aiohttp.ClientSession, url: str, timeout=20):
    try:
        async with session.get(url, timeout=timeout) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                print(f"[WARN] {resp.status} for {url}")
                return None
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"[ERROR] Exception fetching {url}: {e}")
        return None

# ---------- Crawler ----------
async def crawl_and_save_raw(mode="top", limit=None):
    print(f"=== STAGE 1: CRAWL (mode={mode}) ===")
    db = await aiosqlite.connect(RAW_DB)
    await db.execute(RAW_CREATE_SQL)
    await db.commit()

    timeout = aiohttp.ClientTimeout(total=30)
    headers = {"User-Agent": "hn-crawler/2.0 (+your-email@example.com)"}

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        ids = []
        if mode == "top":
            ids = await fetch_json(session, TOP_STORIES)
            if not ids:
                return 0
            if limit:
                ids = ids[:limit]
            else:
                ids = ids[:DEFAULT_TOP_LIMIT]
        elif mode == "archive":
            max_id = await fetch_json(session, MAXITEM)
            if not max_id:
                return 0
            n = limit or DEFAULT_ARCHIVE_LIMIT
            start = max_id - n
            ids = list(range(start, max_id + 1))
        else:
            print("[ERROR] Неизвестный режим:", mode)
            return 0

        print(f"[Crawler] Загружаем {len(ids)} ID...")

        q = asyncio.Queue()
        for sid in ids:
            await q.put(sid)

        async def worker(wid):
            while True:
                sid = await q.get()
                try:
                    item_url = ITEM_URL.format(id=sid)
                    item = await fetch_json(session, item_url)
                    if item is None:
                        continue
                    raw_text = json.dumps(item, ensure_ascii=False)
                    await db.execute(INSERT_RAW_SQL, (item.get("id"), raw_text, time.time()))
                    await db.commit()
                    print(f"[Crawl W{wid:02d}] saved id={sid}")
                    await asyncio.sleep(REQUEST_DELAY)
                finally:
                    q.task_done()

        workers = [asyncio.create_task(worker(i)) for i in range(CONCURRENT_REQUESTS)]
        await q.join()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    await db.close()
    print(f"=== CRAWL finished: raw data saved to {RAW_DB} ===\n")
    return len(ids)

# ---------- Cleaner ----------
async def parse_clean_and_save():
    print("=== STAGE 2: CLEAN ===")
    raw_db = await aiosqlite.connect(RAW_DB)
    clean_db = await aiosqlite.connect(CLEAN_DB)
    await clean_db.execute(CLEAN_CREATE_SQL)
    await clean_db.commit()

    async with raw_db.execute("SELECT id, raw_json FROM raw_items") as cursor:
        rows = await cursor.fetchall()
        print(f"[Cleaner] Найдено {len(rows)} raw записей для обработки.")
        count = 0
        for rid, raw_json in rows:
            try:
                item = json.loads(raw_json)
            except Exception:
                continue
            title = item.get("title", "") or ""
            raw_text = item.get("text", "") or ""
            cleaned = clean_html_fragment(title + " " + raw_text)
            await clean_db.execute(INSERT_CLEAN_SQL, (
                item.get("id"),
                item.get("type"),
                item.get("by"),
                item.get("time"),
                title,
                raw_text,
                cleaned,
                item.get("url"),
                time.time()
            ))
            count += 1
            if count % 100 == 0:
                await clean_db.commit()
        await clean_db.commit()

    await raw_db.close()
    await clean_db.close()
    print(f"=== CLEAN finished: {count} записей сохранено в {CLEAN_DB} ===\n")
    return count

# ---------- Inspect ----------
def print_examples(n=SAMPLE_PRINT):
    import sqlite3
    conn = sqlite3.connect(CLEAN_DB)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM hn_items")
    total = cur.fetchone()[0]
    print(f"=== ИТОГ: {total} записей в {CLEAN_DB} ===")
    cur.execute("SELECT id, title, substr(cleaned_text,1,120) FROM hn_items ORDER BY fetched_at DESC LIMIT ?", (n,))
    rows = cur.fetchall()
    for r in rows:
        print(r)
    conn.close()

# ---------- Pipeline ----------
async def run_pipeline(mode="top", limit=None):
    fetched = await crawl_and_save_raw(mode=mode, limit=limit)
    if fetched == 0:
        print("Ничего не скачано.")
        return
    await parse_clean_and_save()
    print_examples()

if __name__ == "__main__":
    print("Starting pipeline...")
    print("Choose mode:\n Mode: top, limit: 500\n Mode: archive, limit: 100000")
    a = input("Enter mode: ")
    if a == "top":
        asyncio.run(run_pipeline(mode="top", limit=500))
    elif a == "archive":
        asyncio.run(run_pipeline(mode="archive", limit=100000))
    else:
        print("Invalid mode")

