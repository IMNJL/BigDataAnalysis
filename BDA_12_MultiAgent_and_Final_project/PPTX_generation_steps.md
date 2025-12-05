# PPTX Generation — Experiment Steps

This document lists concise, repeatable steps to collect experiment outputs and assemble a presentation (`.pptx`). You will run the commands locally and produce the final slides manually or using `python-pptx`.

## 1. Gather outputs & assets
- Ensure experiment outputs are available in `outputs/` (reports, stats JSON, plots, flowchart PNG, `image.png` mind map).
- Note important files:
  - `BDA_11_Agent_AutoGPT/outputs/` — auto/manual reports, plots
  - `BDA_12_MultiAgent_and_Final_project/outputs/` — RAG outputs, triplets, cypher, etc.
  - `mcp_stubs/` — `stubs.py` (diagram screenshots), `diagrams/system_architecture.puml`
  - `/tmp/<artifact_id>` — original uploaded CSV (if needed)

## 2. Recommended environment & packages
- Optional Python helper (if automating slides):

```bash
python -m venv .venv
source .venv/bin/activate
pip install python-pptx Pillow
```

- If you use plots saved as PNG, Pillow helps embed them.

## 3. Preprocess visuals
- Open and verify plots (PNG/SVG). Resize to 1024×768 for slide clarity if needed.
- Generate/ export PlantUML diagram PNG or SVG:
  - Use PlantUML or online renderer to export `diagrams/system_architecture.puml` → `architecture.png`.
- Collect the mind map image `image.png` used by `generate_docx.py`.

## 4. Slide order (suggested)
1. Title slide: Project name, authors, date
2. One-line summary: Goal, dataset, approach
3. Architecture slide: `architecture.png` (PlantUML)
4. Pipeline slide: MCP stubs, Ingest → RAG → Report (flowchart)
5. Data snapshot: show head of CSV or key stats table
6. Retrieval demo: top-k examples (use table or bullet list)
7. Metrics / Comparison: auto vs manual (bar chart or table)
8. Examples & artifacts: embed 2–3 plots and mind map
9. Limitations & next steps
10. Appendix: commands to reproduce & environment

## 5. Speaker notes (per slide)
- Keep each slide notes short (2–4 bullets): what the slide shows, why it matters, one takeaway.

## 6. Optional: Automate slide creation (snippet)
- Minimal `python-pptx` example to add a title and image:

```python
from pptx import Presentation
from pptx.util import Inches
prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[5])
left = top = Inches(1)
slide.shapes.add_picture('architecture.png', left, top, width=Inches(8))
prs.save('AutoGPT_experiment_presentation.pptx')
```

## 7. Quality checklist before export
- [ ] All images load and are legible on a projector (test at 1024×768)
- [ ] Text is concise and consistent (fonts & sizes)
- [ ] Each slide has speaker notes with the one key message
- [ ] Data sources and timestamps are shown (where relevant)
- [ ] Exported file opens on a second machine (sanity check)

## 8. Export & deliver
- Save as `AutoGPT_experiment_presentation.pptx` and copy to `deliverables/` or `outputs/`.
- Optionally export PDF from PowerPoint for sharing.

## 9. Troubleshooting
- If an image fails to embed, re-save as PNG using an image editor.
- For long tables, include only top rows and link the full CSV in appendix.
