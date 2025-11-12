# Label Studio setup for Amazon Reviews

This short guide shows how to: create a Label Studio project, import `train.csv`, label reviews, and export labels for use with Snorkel.

1. Install Label Studio (see `requirements.txt`) or run with Docker:

   docker run -it -p 8080:8080 -v "$(pwd)/data":/data heartexlabs/label-studio:latest label-studio start --init

2. Create a new project in the web UI.
   - Project type: Text classification
   - Import data: Upload `train.csv` (it must contain a `text` column). You can upload from `BDA_4_Autolabelling_Snorkel/train.csv`.

3. Configure labeling interface
   - Use two labels: `positive`, `negative`.
   - Use a short instruction: "Select the sentiment expressed in the review text."

4. Start labeling. Label a representative subset (e.g., 500-2,000 items). Export as CSV (the export contains a `label` column or JSON that can be converted to CSV).

5. Use exported labels as a small gold/dev set to evaluate Snorkel LabelModel or to supervise labeling functions.

6. Export and save to `BDA_4_Autolabelling_Snorkel/labelstudio_export.csv` and run `snorkel_pipeline.py` to combine weak labels and train.

Notes:
- For large-scale labeling, consider Label Studio Teams or a managed setup.
- You can also use Label Studio's REST API to import/export programmatically.
