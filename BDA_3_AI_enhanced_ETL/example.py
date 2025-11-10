# Creating promotional images for the pipeline (Crawler -> PDF download -> Extract -> Cleaner -> Report generator)
# This script will:
# 1. Create screenshots/ directory (if not exists)
# 2. Generate simple promo images (one per pipeline step) and a combined banner
# 3. Save images to screenshots/ and list saved files
# Note: Images are simple programmatic graphics created with PIL and are suitable for use as "photos" / promo graphics.
from PIL import Image, ImageDraw, ImageFont
import os, textwrap

OUTDIR = "data/screenshots"
os.makedirs(OUTDIR, exist_ok=True)

# helper to draw rounded rectangle box with label and optional icon-like shape
def draw_step_image(label, color, filename, size=(800,600)):
    img = Image.new("RGB", size, (255,255,255))
    draw = ImageDraw.Draw(img)
    w,h = size
    padding = 60
    box = (padding, h//4, w-padding, h*3//4)
    radius = 24
    # rounded rectangle
    draw.rounded_rectangle(box, radius=radius, fill=color, outline=(20,20,20), width=6)
    # small "icon" - a simple glyph for each step based on label keywords
    icon_center = (box[0]+80, (box[1]+box[3])//2)
    if "Crawler" in label:
        # spider-like
        draw.ellipse((icon_center[0]-36, icon_center[1]-36, icon_center[0]+36, icon_center[1]+36), outline=(20,20,20), width=4)
        for a in range(8):
            draw.line((icon_center[0], icon_center[1], icon_center[0]+int(70*__import__("math").cos(a*__import__("math").pi/4)), icon_center[1]+int(70*__import__("math").sin(a*__import__("math").pi/4))), fill=(20,20,20), width=3)
    elif "PDF" in label:
        # paper icon
        p = (icon_center[0]-36, icon_center[1]-50, icon_center[0]+36, icon_center[1]+50)
        draw.rectangle(p, outline=(20,20,20), width=4)
        draw.text((p[0]+8,p[1]+6), "PDF", fill=(20,20,20))
    elif "Extract" in label:
        # document with lines
        p = (icon_center[0]-36, icon_center[1]-50, icon_center[0]+36, icon_center[1]+50)
        draw.rectangle(p, outline=(20,20,20), width=4)
        for i in range(4):
            draw.line((p[0]+8, p[1]+12+ i*18, p[2]-8, p[1]+12+i*18), fill=(20,20,20), width=3)
    elif "Cleaner" in label:
        # funnel
        x,y = icon_center
        draw.polygon([(x-36,y-50),(x+36,y-50),(x+10,y+10),(x+10,y+36),(x-10,y+36),(x-10,y+10)], outline=(20,20,20), fill=None, width=4)
    else:
        # chart
        bx = icon_center[0]-36; by = icon_center[1]+30
        draw.rectangle((bx, by-70, bx+16, by), outline=(20,20,20), width=4)
        draw.rectangle((bx+26, by-50, bx+42, by), outline=(20,20,20), width=4)
        draw.rectangle((bx+52, by-30, bx+68, by), outline=(20,20,20), width=4)
        draw.line((bx, by-70, bx+68, by-70), fill=(20,20,20), width=3)
    # label centered in box
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
    lines = textwrap.wrap(label, width=22)
    text_y = box[1] + 20
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(((w - tw) / 2, text_y), line, fill=(10, 10, 10), font=font)
        text_y += th + 6

    # save
    path = os.path.join(OUTDIR, filename)
    img.save(path)
    return path

steps = [
    ("Crawler", (200,230,255), "01_crawler.png"),
    ("PDF download", (200,255,220), "02_pdf_download.png"),
    ("Extract", (255,240,220), "03_extract.png"),
    ("Cleaner (heuristics/LLM)", (235,225,255), "04_cleaner.png"),
    ("Report generator", (220,235,255), "05_report_generator.png"),
]

saved = []
for label,color,fn in steps:
    p = draw_step_image(label, color, fn)
    saved.append(p)

# create a wide banner combining the five images with arrows
from PIL import ImageOps
imgs = [Image.open(p) for p in saved]
thumb_h = 360
scaled = []
for im in imgs:
    ratio = thumb_h / im.height
    scaled.append(im.resize((int(im.width*ratio), thumb_h)))
spacing = 40
total_w = sum(im.width for im in scaled) + spacing*(len(scaled)-1) + 160
banner = Image.new("RGB", (total_w, thumb_h+120), (255,255,255))
x = 80
y = 40
for i,im in enumerate(scaled):
    banner.paste(im, (x,y))
    x += im.width
    if i < len(scaled)-1:
        # draw arrow to next
        drawb = ImageDraw.Draw(banner)
        ay = y + thumb_h//2
        drawb.line((x, ay, x+spacing-10, ay), fill=(30,30,30), width=10)
        # arrow head
        drawb.polygon([(x+spacing-10, ay-12),(x+spacing+6, ay),(x+spacing-10, ay+12)], fill=(30,30,30))
        x += spacing
# title
try:
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
except Exception:
    title_font = ImageFont.load_default()
draw_banner = ImageDraw.Draw(banner)
title = "Simple Pipeline Architecture"
tw,th = draw_banner.textsize(title, font=title_font)
draw_banner.text(((banner.width-tw)//2, 8), title, fill=(10,10,10), font=title_font)
banner_path = os.path.join(OUTDIR, "banner_pipeline.png")
banner.save(banner_path)
saved.append(banner_path)

# List files created
print("Created files:")
for s in saved:
    print(s)

# Also create a minimal README promo snippet and a simple experiment report template saved as markdown
readme = f"""# Pipeline Promo Images\n\nCreated promotional images for pipeline steps and a combined banner.\n\nFiles in screenshots/: \n{os.listdir(OUTDIR)}\n\nUse these images in your documentation or insert them into your experimental report.\n"""
with open(os.path.join(OUTDIR, "README_PROMO.md"), "w") as f:
    f.write(readme)

report_template = """# Experimental Report Template\n\n## 1. Experiment Overview\n- Pipeline: Crawler -> PDF download -> Extract -> Cleaner (heuristics/LLM) -> Report generator\n- Dataset: PDFs downloaded into `data/raw_pdfs`\n\n## 2. Steps\n1. Run crawler to download PDFs into `data/raw_pdfs`.\n   - Prompt: (seed URL)\n   - Response: PDF files downloaded into `data/raw_pdfs`\n2. Run cleaner to extract text and comments from PDFs.\n   - Prompt: (run cleaner)\n   - Response: JSON files in `data/processed`\n\n## 3. Processed files and example\n- Source: Amazon-2024-Annual-Report.pdf\n- Full text length: 319,924\n- Top comments: (see processed JSON)\n\n## 4. Results and Discussion\n- Summarize extraction accuracy, comment quality, issues encountered.\n\n## 5. Screenshots\nInsert images from `screenshots/` here in order: banner_pipeline.png, 01_crawler.png, 02_pdf_download.png, 03_extract.png, 04_cleaner.png, 05_report_generator.png\n\n## 6. Conclusion\n- Next steps\n"""
with open(os.path.join(OUTDIR, "REPORT_TEMPLATE.md"), "w") as f:
    f.write(report_template)

print("\\nGenerated README_PROMO.md and REPORT_TEMPLATE.md in screenshots/")
