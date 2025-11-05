#!/usr/bin/env python3
"""
ДЕМО: Рабочие примеры Hacker News Crawler
"""

import asyncio
from agent import run_agent

async def demo_single_url():
    """Демо 1: Обработка одной URL"""
    print("=" * 50)
    print("🎯 ДЕМО 1: Обработка одной URL")
    print("=" * 50)
    
    # Тестируем с простой стабильной страницей
    url = "https://httpbin.org/html"
    print(f"🌐 URL: {url}")
    
    await run_agent(url, "url")
    print("✅ Готово!\n")

async def demo_topic_search():
    """Демо 2: Поиск по теме"""
    print("=" * 50)
    print("🔍 ДЕМО 2: Поиск статей по теме 'javascript'")
    print("=" * 50)
    
    topic = "javascript"
    print(f"📝 Тема: {topic}")
    
    await run_agent(topic, "topic")
    print("✅ Готово!\n")

async def show_results():
    """Показываем результаты"""
    print("=" * 50)
    print("📊 РЕЗУЛЬТАТЫ")
    print("=" * 50)
    
    from inspector import print_examples
    print_examples(n=5)  # Показываем только 5 последних записей

async def main():
    """Главная функция демо"""
    print("🚀 HACKER NEWS CRAWLER - ДЕМОНСТРАЦИЯ")
    print("Этот скрипт покажет, как работает crawler\n")
    
    # Запускаем демо
    await demo_single_url()
    await demo_topic_search()
    await show_results()
    
    print("=" * 50)
    print("🎉 ВСЕ ДЕМО ЗАВЕРШЕНЫ!")
    print("=" * 50)
    print("📁 Данные сохранены в:")
    print("   - hn_raw.db (сырые данные)")
    print("   - hn_cleaned.db (очищенные данные)")

if __name__ == "__main__":
    asyncio.run(main())
