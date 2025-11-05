# crawler.py
import aiohttp
from cleaner import clean_html_fragment
import asyncio

async def fetch_json(session, url, timeout=20):
    try:
        async with session.get(url, timeout=timeout) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                print(f"[WARN] {resp.status} for {url}")
                return None
    except Exception as e:
        print(f"[ERROR] Exception fetching {url}: {e}")
        return None

async def fetch_from_url(session, url):
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                html_text = await resp.text()
                return clean_html_fragment(html_text)
    except Exception as e:
        print(f"[ERROR] fetch_from_url {url}: {e}")
    return ""

async def fetch_from_topic(topic, limit=50):
    search_url = f"https://hn.algolia.com/api/v1/search?query={topic}&tags=story&hitsPerPage={limit}"
    async with aiohttp.ClientSession() as session:
        data = await fetch_json(session, search_url)
        urls = [hit["url"] for hit in data.get("hits", []) if hit.get("url")]
        return urls

async def crawl_urls(urls, concurrent_requests=8):
    async with aiohttp.ClientSession() as session:
        q = asyncio.Queue()
        for u in urls:
            await q.put(u)

        results = [None] * len(urls)

        async def worker(wid):
            while not q.empty():
                url = await q.get()
                idx = urls.index(url)
                results[idx] = await fetch_from_url(session, url)
                q.task_done()

        workers = [asyncio.create_task(worker(i)) for i in range(concurrent_requests)]
        await q.join()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        return results
