# cleaner.py
import html
import re
from bs4 import BeautifulSoup
from typing import Optional

def clean_html_fragment(raw_html: Optional[str]) -> str:
    """Удалить HTML, привести к нижнему регистру и очистить лишнее."""
    if not raw_html:
        return ""
    un = html.unescape(raw_html)
    soup = BeautifulSoup(un, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", "[email_removed]", text)
    text = re.sub(r"[\$€£¥]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return text.lower()
