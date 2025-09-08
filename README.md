# Multi-Dataset Bankruptcy Prediction (>100k rows)

## Project Overview
This project predicts bankruptcy (1) vs non-bankruptcy (0) across three datasets: Poland (multi-horizon), Taiwan, and US.  
It uses gradient-boosted trees (LightGBM), handles class imbalance via SMOTE/class-weighting, and applies calibration + threshold tuning for reliable probability outputs.

### Key Features
- Unified dataset with `dataset_source` and `horizon_years` annotations.
- Stratified 80/20 splits per dataset; group-aware for Poland if a firm ID exists.
- Preprocessing: missing value imputation, optional standardization.
- Primary model: LightGBM classifier (XGBoost / CatBoost optional for benchmarking).
- Accuracy-boosting: class weights, SMOTE, early stopping, threshold tuning.
- Calibration: Platt scaling / sigmoid calibration.
- Reproducibility: fixed random seed, requirements.txt, runbook.txt.
- Evaluation: ROC-AUC, PR-AUC, F1, Accuracy, Precision/Recall, Confusion Matrix, calibration plots.


---

## Repository Structure
├── Multi Dataset Bankruptcy Prediction / 
   └──bankcruptcy_outputs /   
           └──  data/  
           └──  eda/  
           └──  models/  
           └──  reports/       
   └── requirements.txt
   └── runbook.txt   
   └── bankruptcy_prediction.ipynb  
   └── README.md  
   └── INSTALLATION.md

---

## Usage

1. **Install dependencies**:
```bash
pip install -r requirements.txt
Set directories in the notebook:

BASE_DIR = "data/raw"
DATA_DIR = "data"
EDA_DIR = "eda"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
````

Load the final model for predictions:
````
from joblib import load
bm = load("models/bankruptcy_model.pkl")
preds = bm.predict(new_data)          
probs = bm.predict_proba(new_data)
````    
Outputs  
models/final_artifact_calibrated.joblib — full pipeline including preprocessor, LightGBM classifier, and calibrator.

models/bankruptcy_model.pkl — picklable wrapper with thresholding for deployment.

eda/ — visualizations and descriptive statistics.

reports/ — split sizes, CV results, evaluation metrics.

requirements.txt — minimal reproducible environment.

runbook.txt — step-by-step instructions.


## Notes

status_label from the US dataset is dropped after creating y to prevent leakage.

Early stopping and calibration improve generalization.

SMOTE is used optionally; CV results determine the final setting.

Random seeds ensure reproducibility across runs.



References
LightGBM documentation: https://lightgbm.readthedocs.io/

Imbalanced-learn: https://imbalanced-learn.org/

Scikit-learn calibration: https://scikit-learn.org/stable/modules/calibration.html