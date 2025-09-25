# db.py
import aiosqlite
import time

RAW_DB = "hn_raw.db"
CLEAN_DB = "hn_cleaned.db"

RAW_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS raw_items (
    id INTEGER PRIMARY KEY,
    raw_text TEXT,
    fetched_at REAL
);
"""

CLEAN_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS hn_items (
    id INTEGER PRIMARY KEY,
    type TEXT,
    by_user TEXT,
    time INTEGER,
    title TEXT,
    raw_text TEXT,
    cleaned_text TEXT,
    url TEXT,
    fetched_at REAL
);
"""

INSERT_RAW_SQL = "INSERT OR REPLACE INTO raw_items (id, raw_text, fetched_at) VALUES (?, ?, ?)"
INSERT_CLEAN_SQL = """
INSERT OR REPLACE INTO hn_items (id, type, by_user, time, title, raw_text, cleaned_text, url, fetched_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

async def save_raw(items):
    db = await aiosqlite.connect(RAW_DB)
    await db.execute(RAW_CREATE_SQL)
    await db.commit()
    for idx, text in enumerate(items):
        await db.execute(INSERT_RAW_SQL, (idx, text, time.time()))
    await db.commit()
    await db.close()

async def save_clean(items, urls, title="topic"):
    db = await aiosqlite.connect(CLEAN_DB)
    await db.execute(CLEAN_CREATE_SQL)
    await db.commit()
    for idx, text in enumerate(items):
        await db.execute(INSERT_CLEAN_SQL, (
            idx, "page", "agent", int(time.time()), title,
            text, text, urls[idx], time.time()
        ))
    await db.commit()
    await db.close()
