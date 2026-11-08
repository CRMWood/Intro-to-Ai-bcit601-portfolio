"""Shared fixtures. pytest finds this file automatically."""
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

import main

BASE_DIR = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def client():
    # "with" runs the app's startup code, so the model is loaded before any test runs
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def model():
    return joblib.load(BASE_DIR/ "Model" / "Final_Model" / "request_text_model-CRW0262_BCIT601_request_classifier.joblib")


@pytest.fixture(scope="module")
def vectoriser():
    return joblib.load(BASE_DIR / "Model" / "Final_Model" / "request_text_vectoriser-CRW0262_BCIT601_request_classifier.joblib")
