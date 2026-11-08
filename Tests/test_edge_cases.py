"""Edge cases: unusual input, boundaries and things going wrong."""
import numpy as np
import pytest

import main


# ---------- Invalid input: the client made a mistake, so expect 422 ----------

INVALID_BODIES = [
    ({}, "field missing"),
    ({"wrong_field": "drain cover missing"}, "wrong field name"),
    ({"text": ""}, "empty string"),
    ({"text": "   "}, "spaces only"),
    ({"text": "\n\t \n"}, "whitespace characters only"),
    ({"text": None}, "null"),
    ({"text": 123}, "number instead of text"),
    ({"text": ["drain", "cover", "missing"]}, "list instead of text"),
    ({"text": {"message": "drain cover missing"}}, "object instead of text"),
]


@pytest.mark.parametrize("body, description", INVALID_BODIES)
def test_invalid_body_is_rejected_with_422(client, body, description, results_log):
    response = client.post("/predict", json=body)
    results_log(f"predict invalid: {description}", response)
    assert response.status_code == 422, description


def test_broken_json_is_rejected_with_422(client, results_log):
    response = client.post("/predict", content="{text: oops",
                           headers={"Content-Type": "application/json"})
    results_log("predict invalid: broken JSON", response)
    assert response.status_code == 422


def test_plain_text_body_is_rejected(client, results_log):
    response = client.post("/predict", content="pothole on my street",
                           headers={"Content-Type": "text/plain"})
    results_log("predict invalid: plain text body", response)
    assert response.status_code == 422


# ---------- Boundaries: test just below, on and just above the limit ----------

def test_one_character_is_accepted(client, results_log):
    response = client.post("/predict", json={"text": "a"})
    results_log("boundry: 1 character", response)
    assert response.status_code == 200


def test_exactly_the_maximum_length_is_accepted(client, results_log):
    text = "printer " * 250            # 8 x 250 = 2000 characters
    assert len(text) == main.MAX_TEXT_LENGTH
    response = client.post("/predict", json={"text": "a"})
    results_log("boundry: exactly 2000 characters", response)
    assert response.status_code == 200


def test_one_character_over_the_maximum_is_rejected(client, results_log):
    text = "x" * (main.MAX_TEXT_LENGTH + 1)
    response = client.post("/predict", json={"text": text})
    results_log("boundry: 2001 characters", response)
    assert response.status_code == 422


# ---------- Unusual but valid text: must not crash ----------

UNUSUAL_TEXT = [
    ("Kia ora, There is a lot of rubish spilling out of the bins. Ngā mihi", "te reo Māori greeting with a macron"),
    ("You still haven't replaced the drain cover 😡", "emoji"),
    ("<script>alert('hi')</script> Recycling bins", "HTML tags"),
    ("'; DROP TABLE tickets; --", "SQL-like text"),
    ("12345 67890", "numbers only"),
    ("!!!???", "punctuation only"),
    ("Brak pokrywy studzienki stwarza zagrożenie w pobliżu płynącej wody.", "a different language"),
    ("MISSING DRAIN COVER!!!", "capital letters"),
    ("drain\ncover\r\nmissing", "line breaks inside the text"),
    ("  No drain cover  ", "spaces around the text"),
    ("a\u0000b drain cover", "a null character"),
]


@pytest.mark.parametrize("text, description", UNUSUAL_TEXT)
def test_unusual_text_gets_a_normal_answer(client, results_log, text, description):
    response = client.post("/predict", json={"text": text})
    results_log(f"unusual text: {description}", response)
    assert response.status_code == 200, description
    assert set(response.json()) == {"category", "confidence", "needs_review", "review_reason"}


# ---------- Risk flags: the API must warn when it is only guessing ----------

@pytest.mark.parametrize("text", ["zzzz qqqq xxxx", "12345", "!!!???", "drin coer missng"])
def test_text_with_no_known_words_needs_review(client, results_log, text):
    response = client.post ("/predict", json={"text": text})
    results_log(f"risk flag: no known words ({text})", response)
    body = response.json()
    assert body["needs_review"] is True
    assert body["review_reason"] == "no_known_words"


def test_vague_text_needs_review_because_confidence_is_low(client, results_log):
    response = client.post("/predict", json={"text": "it is cracked and suggests issue"})
    results_log(f"risk flag: vague text", response)
    body = response.json()
    assert body["needs_review"] is True
    assert body["review_reason"] == "low_confidence"
    assert body["confidence"] < main.CONFIDENCE_THRESHOLD


# ---------- Failure modes: things go wrong inside the service ----------

def test_predict_returns_503_when_the_model_is_not_loaded(client, results_log, monkeypatch):
    # monkeypatch swaps something out for this one test, then puts it back
    monkeypatch.setattr(main, "artefacts", {})
    response = client.post("/predict", json={"text": "drain cover"})
    results_log("failure: health with no model", response)
    assert response.status_code == 503


def test_health_returns_503_when_the_model_is_not_loaded(client, results_log, monkeypatch):
    monkeypatch.setattr(main, "artefacts", {})
    response = client.get("/health")
    results_log("failure: health with no model", response)    
    assert client.get("/health").status_code == 503


class BrokenModel:
    """A pretend model that always fails, with an error message that must stay private."""
    classes_ = ["a", "b"]

    def predict_proba(self, features):
        raise RuntimeError("secret internal detail: /home/admin/model.joblib")


def test_internal_failure_returns_500_without_leaking_details(client, results_log, monkeypatch):
    monkeypatch.setattr(main, "artefacts", {"model": BrokenModel(),
                                            "vectoriser": main.artefacts["vectoriser"]})
    response = client.post("/predict", json={"text": "drain cover"})
    results_log("failure: internal error", response)
    assert response.status_code == 500
    assert "secret internal detail" not in response.text
    assert "Traceback" not in response.text


# ---------- A fake model gives exact control over the threshold rule ----------

class FakeModel:
    """A pretend model that returns fixed probabilities, so the threshold rule can be tested exactly."""
    classes_ = ["waste", "water", "safety", "road"]

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, features):
        return np.array([self.probabilities])


def test_confidence_just_below_the_threshold_needs_review(client, results_log, monkeypatch, vectoriser):
    fake = FakeModel([0.10, 0.40, 0.25, 0.25])            # the top score is only 0.40
    monkeypatch.setattr(main, "artefacts", {"model": fake, "vectoriser": vectoriser})
    response = client.post("/predict", json={"text": "missing drain cover"})
    results_log("threshold: confidence 0.40 (below 0.50)", response)
    body = response.json()
    assert body["category"] == "water"
    assert body["confidence"] == 0.4
    assert body["needs_review"] is True
    assert body["review_reason"] == "low_confidence"


def test_confidence_at_the_threshold_does_not_need_review(client, results_log, monkeypatch, vectoriser):
    fake = FakeModel([0.50, 0.30, 0.10, 0.10])            # exactly on the threshold: not below it
    monkeypatch.setattr(main, "artefacts", {"model": fake, "vectoriser": vectoriser})
    response = client.post("/predict", json={"text": "missing drain cover"})
    results_log("threshold: confidence 0.50 (on the threshold)",response)
    body = response.json()
    assert body["confidence"] == main.CONFIDENCE_THRESHOLD
    assert body["needs_review"] is False
