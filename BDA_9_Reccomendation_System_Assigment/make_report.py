# ...existing code...
import os
import argparse
from docx import Document

PROJECT_ROOT = os.path.dirname(__file__)
OUTPUTS = os.path.join(PROJECT_ROOT, "outputs")
REPORT_PATH = os.path.join(PROJECT_ROOT, "BDA_9_Recommendation_report.docx")

def build_report(user_id):
    result_file = os.path.join(OUTPUTS, f"results_user_{user_id}.txt")
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Run run/recommend_user.py first. Missing {result_file}")
    with open(result_file, "r", encoding="utf-8") as f:
        content = f.read()
    doc = Document()
    doc.add_heading("BDA_9_Recommendation - Automated Report", level=1)
    doc.add_paragraph("Generated outputs for user id: " + str(user_id))
    doc.add_heading("Recommendations and Profile", level=2)
    doc.add_paragraph(content)
    doc.add_heading("Notes", level=2)
    doc.add_paragraph("This report was auto-generated. The LLM tag generation was simulated if --simulate-llm was used.")
    doc.save(REPORT_PATH)
    print(f"Report written to {REPORT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=int, default=50)
    args = parser.parse_args()
    build_report(args.user)
# ...existing code...