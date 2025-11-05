#!/usr/bin/env python3
"""
Простой рабочий пример использования Hacker News crawler
"""

import asyncio
from agent import run_agent

async def test_crawler():
    """Тестируем crawler с простым запросом"""
    print("🚀 Запускаем тест crawler...")
    
    # Используем простой запрос, который точно найдет результаты
    topic = "python"
    
    print(f"📝 Ищем статьи по теме: '{topic}'")
    
    try:
        await run_agent(topic, "topic")
        print("✅ Crawler успешно завершен!")
        
        # Показываем результаты
        from inspector import inspect_db
        print("\n📊 Результаты:")
        inspect_db()
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(test_crawler())
