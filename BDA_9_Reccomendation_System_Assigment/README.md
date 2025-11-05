# BDA_9_Recommendation

This project is designed to provide a recommendation system using various machine learning techniques. It includes data processing, feature engineering, model training, and evaluation.

## Project Structure

- **data/**: Contains raw and processed data files.
  - **raw/**: Store raw data files that have not been processed.
  - **processed/**: Store processed data files that are ready for analysis or modeling.
  
- **notebooks/**: Contains Jupyter notebooks for exploratory data analysis.
  - **exploration.ipynb**: Used for exploratory data analysis, visualizations, and initial insights into the dataset.

- **src/**: Contains the source code for the project.
  - **data/**: Functions for loading and preprocessing data.
  - **features/**: Functions for feature engineering.
  - **models/**: Functions for training models and making recommendations.
  - **utils/**: Utility functions used throughout the project.

- **tests/**: Contains unit tests for the project.

## Run steps (macOS)

- Open Terminal and execute:

1. Create project folder and files as above (or copy provided files).
2. Create and activate venv:
```bash
cd /Users/pro/Downloads/BigDataAnalysis/BDA_9_Recommendation
python3 -m venv venv
source venv/bin/activate
```
3. Install requirements:
```bash
pip install -r requirements.txt
```
4. Place MovieLens ml-100k folder at:

```/Users/pro/Downloads/BigDataAnalysis/BDA_9_Recommendation/data/raw/ml-100k (download from https://grouplens.org/datasets/movielens/100k/)```


5. Run (simulate LLM):

```bash
./run.sh 50 --simulate-llm
```

This will:
- produce outputs/results_user_50.txt
- create BDA_9_Recommendation_report.docx in project root


## Usage

1. Load your data using the functions in `src/data/load.py`.
2. Preprocess the data with `src/data/preprocess.py`.
3. Engineer features using `src/features/build_features.py`.
4. Train your model with `src/models/train.py`.
5. Make recommendations using `src/models/recommend.py`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.