#http://127.0.0.1:8000/docs
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Folder that contains this file, so the model files are found no matter where you start the server.
MODEL_LOCATION = Path(__file__).resolve().parent.parent / "Model" / "Final_Model" 

# A dictionary that holds the loaded model and vectoriser while the app is running
artefacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: runs once, before the first request
    artefacts["model"] = joblib.load(MODEL_LOCATION / "request_text_model-CRW0262_BCIT601_request_classifier.joblib")
    artefacts["vectoriser"] = joblib.load(MODEL_LOCATION / "request_text_vectoriser-CRW0262_BCIT601_request_classifier.joblib")
    yield
    # SHUTDOWN: runs once when the server stops
    artefacts.clear()


app = FastAPI(title="Council Request Ticket Classifier", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class PredictResponse(BaseModel):
    category: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/categories")
def get_categories():
    # classes_ is a NumPy array; .tolist() turns it into a normal Python list
    return {"categories": artefacts["model"].classes_.tolist()}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Text made only of spaces passes min_length, so we check for it ourselves
    if request.text.strip() == "":
        raise HTTPException(status_code=422, detail="text must not be blank")

    model = artefacts["model"]
    vectoriser = artefacts["vectoriser"]

    features = vectoriser.transform([request.text])       # 1. text -> numbers
    probabilities = model.predict_proba([request.text])[0]      # 2. one probability per category
    best_index = probabilities.argmax()                   # 3. position of the largest one

    return {
        "category": str(model.classes_[best_index]),
        "confidence": round(float(probabilities[best_index]), 3),
    }


# This block lets you start the server with the Run button in VS Code, or with: python main.py
# (You can also start it from a terminal with: fastapi dev main.py)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)
