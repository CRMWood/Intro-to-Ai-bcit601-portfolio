from pathlib import Path

# Project root (this file lives one level below it)
PROJECT_ROOT = Path.cwd()

# Folders
RAW_DIR = PROJECT_ROOT / "Data" / "Raw"
PROCESSED_DIR = PROJECT_ROOT / "Data" / "Processed"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
MODEL_DIR = PROJECT_ROOT / "Model"
TEST_DIR = PROJECT_ROOT / "Test"


# Creates folder if the folder doesn't exist as Github doesn't track empty folders
for folder in (PROCESSED_DIR, OUTPUTS_DIR, MODEL_DIR, TEST_DIR):
    folder.mkdir(parents=True, exist_ok=True)

