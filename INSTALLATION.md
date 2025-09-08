## Installation Guide - Multi-Dataset Bankruptcy Prediction

This guide provides instructions to set up the environment and dependencies for the bankruptcy prediction project. The project uses Python 3.x and standard data science libraries with gradient-boosted tree models (LightGBM, XGBoost, CatBoost).

## 1. System Requirements

Python: 3.9+ (tested on 3.10)

OS: Windows, Linux, or macOS

Memory: ≥8 GB RAM recommended

Disk: ≥2 GB free for datasets, models, and reports



## 2. Create a Virtual Environment (Recommended)

Using venv:
````
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
````
## 3 . Install Dependencies

Install the core dependencies for running the notebook and reproducing results:
````
pip install --upgrade pip
pip install -r requirements.txt
````

requirements.txt contains the exact library versions used in the project to ensure reproducibility.


## 4. Directory Structure

Make sure the project directories exist or are created automatically:

BASE_DIR/          # Raw datasets (Poland, Taiwan, US)  
DATA_DIR/          # Combined cleaned dataset  
MODELS_DIR/        # Saved model artifacts  
REPORTS_DIR/       # CV results, evaluation metrics  
EDA_DIR/           # Exploratory analysis outputs


You can modify BASE_DIR and other paths in the notebook as needed.

## 5. Data Placement

Place Poland ARFF files in BASE_DIR:

1year.arff  
2year.arff  
3year.arff  
4year.arff  
5year.arff  

Place Taiwan and US CSVs in BASE_DIR:

taiwan_bankruptcy.csv  
american_bankruptcy.csv

## 6. Running the Notebook

Open the main notebook (bankruptcy_prediction.ipynb) in Jupyter:
````
jupyter notebook
````

Then execute cells sequentially. The runbook (runbook.txt) in OUT_DIR provides a step-by-step reference.

## 7. Reproducibility

Fixed random seeds are used for CV and model training.

Environment versions are captured in requirements.txt.

Final artifacts, including the preprocessor, model, calibrator, and threshold, are saved in:

MODELS_DIR/final_artifact_calibrated.joblib

## 8. Troubleshooting

Missing packages: ensure you installed all dependencies via requirements.txt.

Data not found: verify BASE_DIR paths and filenames.

Memory issues: consider sampling or increasing available RAM.

## 9. Optional Enhancements

GPU acceleration for LightGBM/XGBoost/CatBoost if supported.

Extended EDA by installing optional packages in requirements_full.txt.

Use virtual environment snapshots (conda/pip freeze) to replicate exact setup.

Project Artifacts Reference:

requirements.txt – core dependencies

runbook.txt – step-by-step workflow

final_artifact_calibrated.joblib – saved preprocessor, model, calibrator, and threshold