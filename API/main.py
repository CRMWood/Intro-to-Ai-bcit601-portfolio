import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(__file__).parent.parent / "Model" / "Final_Model"
MODEL_VERSION = "Council_Request-V1"
CONFIDENCE_THRESHOLD = 0.50   # below this, a staff member should check the result
MAX_TEXT_LENGTH = 2000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("council_api")

artefacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: if a file is missing, joblib.load raises an error and the server stops.
    # That is deliberate: stopping with a clear message is safer than running with no model.
    artefacts["model"] = joblib.load(MODEL_DIR / "request_text_model-CRW0262_BCIT601_request_classifier.joblib")
    artefacts["vectoriser"] = joblib.load(MODEL_DIR / "request_text_vectoriser-CRW0262_BCIT601_request_classifier.joblib")

    logger.info("Loaded %s with %d categories", MODEL_VERSION, len(artefacts["model"].classes_))
    yield
    # SHUTDOWN
    artefacts.clear()


app = FastAPI(title="City Council Request Classifier", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)


class PredictResponse(BaseModel):
    category: str
    confidence: float
    needs_review: bool
    review_reason: str | None = None


def model_is_loaded():
    if "model" in artefacts and "vectoriser" in artefacts:
        return True
    return False


@app.get("/health")
def health():
    if not model_is_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ok", "model_loaded": True, "model_version": MODEL_VERSION}


@app.get("/categories")
def get_categories():
    if not model_is_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"categories": artefacts["model"].classes_.tolist()}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # 1. The service cannot answer without its model
    if not model_is_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # 2. Text made only of spaces passes min_length, so check it here
    text = request.text.strip()
    if text == "":
        raise HTTPException(status_code=422, detail="text must not be blank")

    model = artefacts["model"]
    vectoriser = artefacts["vectoriser"]

    # 3. Predict. If anything goes wrong, log the details and return a safe message.
    try:
        features = vectoriser.transform([text])
        probabilities = model.predict_proba(features)[0]
    except Exception:
        logger.exception("Prediction failed")   # the full error goes to the log only
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again later.")

    best_index = probabilities.argmax()
    category = str(model.classes_[best_index])
    confidence = round(float(probabilities[best_index]), 3)

    # 4. Decide whether a person should check this result
    needs_review = False
    review_reason = None
    if features.nnz == 0:
        # None of the words are in the vocabulary, so the model is only guessing
        needs_review = True
        review_reason = "no_known_words"
    elif confidence < CONFIDENCE_THRESHOLD:
        needs_review = True
        review_reason = "low_confidence"

    # 5. Log the outcome, but never the ticket text (it may contain personal information)
    logger.info("predict length=%d category=%s confidence=%.3f needs_review=%s",
                len(text), category, confidence, needs_review)

    return {
        "category": category,
        "confidence": confidence,
        "needs_review": needs_review,
        "review_reason": review_reason,
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", reload=True)