# Intelligent Hacker News Crawler

## 📌 Description
Этот проект реализует умный краулер для [Hacker News](https://news.ycombinator.com/),  
который можно использовать в двух режимах:

1. **Классический пайплайн** (`hn_crawler_pipeline.py`)  
   - **Crawl** → скачивает данные (`hn_raw.db`)  
   - **Clean** → очищает текст (HTML, спецсимволы) → сохраняет в `hn_cleaned.db`)  
   - **Inspect** → показывает примеры  

2. **Интеллектуальный агент** (`main.py`)  
   - Принимает **тему (натуральный язык)** или **URL**  
   - Автоматически ищет статьи по теме или скачивает страницу  
   - Чистит текст и сохраняет в базу (`hn_raw.db` + `hn_cleaned.db`)  
   - Позволяет просмотреть примеры результатов  

---

## 📂 Architecture
```
hn-intelligent-crawler/
│── hn_crawler_pipeline.py # классический пайплайн HN
│── main.py # интеллектуальный агент
│── crawler.py # логика скачивания страниц
│── cleaner.py # функции очистки текста
│── db.py # работа с SQLite
│── agent.py # связывает crawl + clean + save
│── inspector.py # просмотр/печать примеров
│── README.md # инструкция
│── requirements.txt # зависимости
│── .gitignore # исключает .db
```


---

## 🚀 Installation
```bash
git clone https://github.com/IMNJL/BigDataAnalysis/hn-intelligent-crawler.git
cd hn-intelligent-crawler
pip install -r requirements.txt
```

## ▶️ Running
1. Классический пайплайн

```bash
python hn_crawler_pipeline.py
```

Скачивает топовые или архивные записи HN, очищает текст, сохраняет в базу и показывает примеры.
2. Интеллектуальный агент
```zsh
python main.py
```

Вводите topic или url, агент автоматически скачивает и очищает данные, затем показывает результаты.
