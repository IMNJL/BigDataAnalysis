"""
Simple crawler for BDA_3_Ai_enhanced_ETL
- Accepts either a Topic text or a Link URL via --input
- If input looks like a URL and points to a PDF, downloads it
- If input is a webpage URL, downloads page and extracts links to PDFs and downloads them
- Saves PDFs to data/raw_pdfs/

This is a lightweight, offline-friendly scaffold. For "topic" input you should provide seed URLs or extend with a search API.
"""
import os
import argparse
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

OUT_DIR = os.path.join(os.path.dirname(__file__), 'data', 'raw_pdfs')
os.makedirs(OUT_DIR, exist_ok=True)


def download_file(url, out_dir=OUT_DIR, timeout=20):
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        filename = os.path.basename(urlparse(url).path) or 'downloaded.pdf'
        if not filename.lower().endswith('.pdf'):
            # try to derive pdf filename
            filename = filename + '.pdf'
        out_path = os.path.join(out_dir, filename)
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(1024 * 32):
                if chunk:
                    f.write(chunk)
        return out_path
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None


def crawl_page_for_pdfs(page_url, out_dir=OUT_DIR):
    print(f"Crawling page: {page_url}")
    try:
        r = requests.get(page_url, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch page {page_url}: {e}")
        return []
    soup = BeautifulSoup(r.text, 'html.parser')
    pdf_links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.lower().endswith('.pdf'):
            pdf_links.append(urljoin(page_url, href))
    downloaded = []
    for url in sorted(set(pdf_links)):
        print(f"Found PDF: {url}")
        p = download_file(url, out_dir=out_dir)
        if p:
            downloaded.append(p)
    return downloaded


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Topic text or URL to crawl')
    parser.add_argument('--outdir', '-o', default=OUT_DIR)
    args = parser.parse_args()

    inp = args.input.strip()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # naive detection: if starts with http(s) treat as URL
    if inp.lower().startswith('http://') or inp.lower().startswith('https://'):
        # if URL points to a PDF
        if inp.lower().endswith('.pdf'):
            p = download_file(inp, out_dir=outdir)
            if p:
                print(f"Downloaded PDF to {p}")
        else:
            downloaded = crawl_page_for_pdfs(inp, out_dir=outdir)
            if not downloaded:
                print("No PDFs found on page.")
            else:
                print(f"Downloaded {len(downloaded)} PDFs to {outdir}")
    else:
        # topic mode: prompt user to provide seed URLs or extend this to call a search API
        print("Input looks like a topic, not a URL. Please provide seed URLs or extend the crawler to use a search API.")
        print("Received topic:", inp)
