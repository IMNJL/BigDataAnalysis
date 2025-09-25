# agent.py
import asyncio
from crawler import fetch_from_topic, crawl_urls
from db import save_raw, save_clean

async def run_agent(input_value, input_type="topic"):
    if input_type == "topic":
        urls = await fetch_from_topic(input_value)
    else:
        urls = [input_value]

    print(f"[Agent] Found {len(urls)} URLs to crawl")
    raw_texts = await crawl_urls(urls)

    await save_raw(raw_texts)
    await save_clean(raw_texts, urls, title=input_value)
    print("[Agent] Finished crawling & cleaning")
