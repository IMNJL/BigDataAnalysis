"""
hn_utils.py
Вспомогательные функции: очистка HTML/текста.
"""

import html
import re
from bs4 import BeautifulSoup
from typing import Optional


def clean_html_fragment(raw_html: Optional[str]) -> str:
    """Удалить HTML, привести к нижнему регистру и очистить лишнее."""
    if not raw_html:
        return ""
    # Декодируем HTML сущности
    un = html.unescape(raw_html)
    # Убираем HTML-теги
    soup = BeautifulSoup(un, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    # Чистим
    text = re.sub(r"\s+", " ", text)                       # пробелы
    text = re.sub(r"\S+@\S+\.\S+", "[email_removed]", text) # e-mail
    text = re.sub(r"[\$€£¥]", " ", text)                   # валютные знаки
    text = re.sub(r"[^\w\s]", " ", text)                   # пунктуация
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()
