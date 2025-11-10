"""
Generate a simple architecture diagram (PNG) for the ETL pipeline and save it to screenshots/architecture.png.
This script uses Pillow to draw boxes and arrows. It's a utility to produce a presentable diagram without
needing external design tools.
"""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(__file__)
SCREENSHOT_DIR = os.path.join(ROOT, 'screenshots')
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
OUT_PATH = os.path.join(SCREENSHOT_DIR, 'architecture.png')

# Canvas
W, H = 1200, 600
bg = (255, 255, 255)
img = Image.new('RGB', (W, H), bg)
d = ImageDraw.Draw(img)

# try to load a default font
try:
    font = ImageFont.truetype('DejaVuSans.ttf', 16)
except Exception:
    font = ImageFont.load_default()

# helper to draw a box with text
def draw_box(x, y, w, h, text):
    rect_fill = (240, 248, 255)
    rect_outline = (60, 90, 150)
    d.rectangle([x, y, x+w, y+h], fill=rect_fill, outline=rect_outline, width=2)
    # center multiline text using textbbox (compatible with newer Pillow)
    try:
        bbox = d.multiline_textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        # fallback: approximate using getsize for older Pillow
        try:
            tw, th = font.getsize(text)
        except Exception:
            tw, th = (w, h)
    tx = x + (w - tw) / 2
    ty = y + (h - th) / 2
    # draw multiline text centered
    d.multiline_text((tx, ty), text, fill=(20, 30, 50), font=font, align='center')

# boxes positions
boxes = [
    (50, 220, 200, 80, 'Input: Topic / Seed URLs'),
    (300, 50, 220, 80, 'Crawler\n(download PDFs)'),
    (300, 320, 220, 80, 'Raw Storage\ndata/raw_pdfs'),
    (600, 120, 260, 100, 'Extractor\n(PDF -> Text)'),
    (920, 120, 220, 100, 'Cleaner / LLM\n(Extract comments)'),
    (600, 320, 260, 100, 'Processed Storage\ndata/processed'),
    (920, 320, 220, 100, 'Report Generator\nDOCX/JSON'),
]

for x,y,w,h,txt in boxes:
    draw_box(x,y,w,h,txt)

# arrows (simple lines with triangle heads)
def arrow(fr, to):
    fx, fy = fr
    tx, ty = to
    d.line([fx, fy, tx, ty], fill=(60,60,60), width=3)
    # small triangle
    # compute simple perpendicular for head
    d.polygon([(tx,ty),(tx-8,ty-6),(tx-8,ty+6)], fill=(60,60,60))

# connect boxes roughly
arrow((250,260),(300,90))   # input -> crawler
arrow((380,130),(600,170))  # crawler -> extractor
arrow((380,180),(300,360))  # crawler -> raw storage
arrow((680,170),(920,170))  # extractor -> cleaner
arrow((760,220),(760,360))  # extractor -> processed storage
arrow((760,360),(920,360))  # processed -> report

# footer text
d.text((20, H-30), 'Generated diagram: ETL pipeline (crawler -> extractor -> cleaner -> report)', fill=(80,80,80), font=font)

img.save(OUT_PATH)
print('Wrote architecture diagram to', OUT_PATH)
