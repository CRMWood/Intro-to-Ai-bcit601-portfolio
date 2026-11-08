"""Shared fixtures. pytest finds this file automatically."""
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

import main

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR  / "Data" / "Processed"
OUTPUTS_DIR = BASE_DIR / "Outputs" 
MODEL_LOCATION = BASE_DIR / "Model" / "Final_Model" 


@pytest.fixture(scope="module")
def client():
    # "with" runs the app's startup code, so the model is loaded before any test runs
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_LOCATION / "request_text_model-CRW0262_BCIT601_request_classifier.joblib")


@pytest.fixture(scope="module")
def vectoriser():
    return joblib.load(MODEL_LOCATION / "request_text_vectoriser-CRW0262_BCIT601_request_classifier.joblib")

#prints a log in the output file showing the testing results
@pytest.fixture(scope="session")
def results_log():
    results_folder = OUTPUTS_DIR / "Check_API_Tests"
    results_folder.mkdir(parents=True, exist_ok=True) #creates the folder if missing
    # creates the results text file, writes to it and ensures that the coding is correct so that it doesn't crash from a strange character
    with open(results_folder / "standard_test_results.txt", "w", encoding="utf-8") as f:

        #defines how the log will look
        def log(label, response):
            request = response.request
            body = request.content.decode() if request.content else "(no body)"
            lines = [
                f"--- {label} ---",
                f"Request:  {request.method} {request.url.path}",
                f"Body:     {body[:100]}{'...' if len(body) > 100 else ''}",
                f"Status:   {response.status_code}",
                f"Response: {response.text[:300]}{'...' if len(response.text) > 300 else ''}",
                "",
            ]
            for line in lines:
                print(line)
                print(line, file=f)

        yield log