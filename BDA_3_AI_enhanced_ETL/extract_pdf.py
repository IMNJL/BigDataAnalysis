"""
Helpers to extract text from PDFs using PyPDF2 and simple cleaning helpers.
"""
import os
from typing import List

try:
    import PyPDF2
except Exception:
    PyPDF2 = None


def extract_text_from_pdf(path: str) -> str:
    if PyPDF2 is None:
        raise ImportError("PyPDF2 is required. Install with `pip install PyPDF2`.")
    text_parts: List[str] = []
    try:
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                try:
                    t = p.extract_text() or ''
                    text_parts.append(t)
                except Exception:
                    continue
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return '\n'.join(text_parts)


def basic_clean_text(text: str) -> str:
    # Normalize whitespace and remove repeated short lines
    import re
    txt = text.replace('\r', '\n')
    txt = re.sub('\n{2,}', '\n\n', txt)
    # remove many whitespace
    txt = re.sub('[ \t]{2,}', ' ', txt)
    # strip leading/trailing spaces on each line
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    return '\n'.join(lines)


def extract_comments_from_text(text: str) -> List[str]:
    # Heuristic: find lines that look like notes/comments or bracketed parentheticals
    import re
    out = []
    for ln in text.splitlines():
        if len(ln) > 10 and (ln.lower().startswith('note') or ln.lower().startswith('comment') or 'note:' in ln.lower()):
            out.append(ln)
        # short parenthetical sentences
        if re.search('\\(.*comment|note|remark.*\\)', ln.lower()):
            out.append(ln)
    return out
