#!/usr/bin/env python3
"""
Супер простой пример - тестируем crawler с одной известной URL
"""

import asyncio
from agent import run_agent

async def simple_test():
    """Простой тест с одной URL"""
    print("🔍 Тестируем crawler с одной URL...")
    
    # Используем известную стабильную URL
    test_url = "https://httpbin.org/html"
    
    print(f"🌐 Тестируем URL: {test_url}")
    
    try:
        await run_agent(test_url, "url")
        print("✅ Тест успешно завершен!")
        
        # Проверяем что данные сохранились
        import aiosqlite
        async with aiosqlite.connect("hn_raw.db") as db:
            async with db.execute("SELECT COUNT(*) FROM raw_items") as cursor:
                count = await cursor.fetchone()
                print(f"📁 Сохранено {count[0]} элементов в raw_items")
        
        async with aiosqlite.connect("hn_cleaned.db") as db:
            async with db.execute("SELECT COUNT(*) FROM hn_items") as cursor:
                count = await cursor.fetchone()
                print(f"📁 Сохранено {count[0]} элементов в hn_items")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(simple_test())
