from pathlib import Path

# Project root (this file lives one level below it)
PROJECT_ROOT = Path.cwd()

# Folders
RAW_FILE = PROJECT_ROOT / "Data" / "Raw"
PROCESSED_FILE = PROJECT_ROOT / "Data" / "Processed"
OUTPUTS_FILE = PROJECT_ROOT / "Outputs"
MODEL_FILE = PROJECT_ROOT / "Model"

# Files
#Data files
RAW_DATA = RAW_FILE / "BCIT601_starter_service_requests_dataset.csv"
CLEAN_DATA = PROCESSED_FILE / "BCIT601_starter_service_requests_dataset_clean.csv"

#Joblib files
MODEL_MODEL = MODEL_FILE / "model - CRW0262_BCIT601_request_classifier- .joblib"
MODEL_VECTORISER = MODEL_FILE / "vectoriser - CRW0262_BCIT601_request_classifier.joblib"



# Make sure output folders exist, as Github won't track empty folders
for folder in (PROCESSED_FILE, OUTPUTS_FILE, MODEL_FILE):
    folder.mkdir(parents=True, exist_ok=True)

