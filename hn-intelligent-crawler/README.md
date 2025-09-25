# Intelligent Hacker News Crawler

## 📌 Описание
Этот проект реализует умный краулер для [Hacker News](https://news.ycombinator.com/), 
работающий как **конвейер**:

1. **Crawl** → скачивает данные (`hn_raw.db`)  
2. **Clean** → очищает текст (HTML, спецсимволы) → сохраняет в `hn_cleaned.db`  
3. **Inspect** → показывает примеры  

## 📂 Структура проекта
```
hn-intelligent-crawler/
│── hn_crawler_pipeline.py # основной пайплайн
│── hn_utils.py # функции очистки текста
│── README.md # инструкция
│── requirements.txt # зависимости
│── .gitignore # исключает .db
```

## 🚀 Установка
```bash
git clone https://github.com/IMNJL/BigDataAnalysis/hn-intelligent-crawler.git
cd hn-intelligent-crawler
pip install -r requirements.txt