"""Edge cases: unusual input, boundaries and things going wrong."""
import numpy as np
import pytest

import main


# ---------- Invalid input: the client made a mistake, so expect 422 ----------

INVALID_BODIES = [
    ({}, "field missing"),
    ({"wrong_field": "printer jammed"}, "wrong field name"),
    ({"text": ""}, "empty string"),
    ({"text": "   "}, "spaces only"),
    ({"text": "\n\t \n"}, "whitespace characters only"),
    ({"text": None}, "null"),
    ({"text": 123}, "number instead of text"),
    ({"text": ["printer", "jammed"]}, "list instead of text"),
    ({"text": {"message": "printer jammed"}}, "object instead of text"),
]


@pytest.mark.parametrize("body, description", INVALID_BODIES)
def test_invalid_body_is_rejected_with_422(client, body, description):
    response = client.post("/predict", json=body)
    assert response.status_code == 422, description


def test_broken_json_is_rejected_with_422(client):
    response = client.post("/predict", content="{text: oops",
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_plain_text_body_is_rejected(client):
    response = client.post("/predict", content="printer jammed",
                           headers={"Content-Type": "text/plain"})
    assert response.status_code == 422


# ---------- Boundaries: test just below, on and just above the limit ----------

def test_one_character_is_accepted(client):
    assert client.post("/predict", json={"text": "a"}).status_code == 200


def test_exactly_the_maximum_length_is_accepted(client):
    text = "printer " * 250            # 8 x 250 = 2000 characters
    assert len(text) == main.MAX_TEXT_LENGTH
    assert client.post("/predict", json={"text": text}).status_code == 200


def test_one_character_over_the_maximum_is_rejected(client):
    text = "x" * (main.MAX_TEXT_LENGTH + 1)
    assert client.post("/predict", json={"text": text}).status_code == 422


# ---------- Unusual but valid text: must not crash ----------

UNUSUAL_TEXT = [
    ("Kia ora, my wifi is down. Ngā mihi", "te reo Māori greeting with a macron"),
    ("my printer 🖨️ is jammed 😡", "emoji"),
    ("<script>alert('hi')</script> printer jammed", "HTML tags"),
    ("'; DROP TABLE tickets; --", "SQL-like text"),
    ("12345 67890", "numbers only"),
    ("!!!???", "punctuation only"),
    ("Il mio computer non si accende", "a different language"),
    ("PRINTER JAMMED!!!", "capital letters"),
    ("printer\njammed\r\nagain", "line breaks inside the text"),
    ("  printer jammed  ", "spaces around the text"),
    ("a\u0000b printer jammed", "a null character"),
]


@pytest.mark.parametrize("text, description", UNUSUAL_TEXT)
def test_unusual_text_gets_a_normal_answer(client, text, description):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200, description
    assert set(response.json()) == {"category", "confidence", "needs_review", "review_reason"}


# ---------- Risk flags: the API must warn when it is only guessing ----------

@pytest.mark.parametrize("text", ["zzzz qqqq xxxx", "12345", "!!!???", "prnter jamed agin"])
def test_text_with_no_known_words_needs_review(client, text):
    body = client.post("/predict", json={"text": text}).json()
    assert body["needs_review"] is True
    assert body["review_reason"] == "no_known_words"


def test_vague_text_needs_review_because_confidence_is_low(client):
    body = client.post("/predict", json={"text": "it is not working please help"}).json()
    assert body["needs_review"] is True
    assert body["review_reason"] == "low_confidence"
    assert body["confidence"] < main.CONFIDENCE_THRESHOLD


# ---------- Failure modes: things go wrong inside the service ----------

def test_predict_returns_503_when_the_model_is_not_loaded(client, monkeypatch):
    # monkeypatch swaps something out for this one test, then puts it back
    monkeypatch.setattr(main, "artefacts", {})
    response = client.post("/predict", json={"text": "printer jammed"})
    assert response.status_code == 503


def test_health_returns_503_when_the_model_is_not_loaded(client, monkeypatch):
    monkeypatch.setattr(main, "artefacts", {})
    assert client.get("/health").status_code == 503


class BrokenModel:
    """A pretend model that always fails, with an error message that must stay private."""
    classes_ = ["A", "B"]

    def predict_proba(self, features):
        raise RuntimeError("secret internal detail: /home/admin/model.joblib")


def test_internal_failure_returns_500_without_leaking_details(client, monkeypatch):
    monkeypatch.setattr(main, "artefacts", {"model": BrokenModel(),
                                            "vectoriser": main.artefacts["vectoriser"]})
    response = client.post("/predict", json={"text": "printer jammed"})
    assert response.status_code == 500
    assert "secret internal detail" not in response.text
    assert "Traceback" not in response.text


# ---------- A fake model gives exact control over the threshold rule ----------

class FakeModel:
    """A pretend model that returns fixed probabilities, so the threshold rule can be tested exactly."""
    classes_ = ["Hardware", "Printing", "Wi-Fi and network"]

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, features):
        return np.array([self.probabilities])


def test_confidence_just_below_the_threshold_needs_review(client, monkeypatch, vectoriser):
    fake = FakeModel([0.40, 0.35, 0.25])            # the top score is only 0.40
    monkeypatch.setattr(main, "artefacts", {"model": fake, "vectoriser": vectoriser})
    body = client.post("/predict", json={"text": "printer jammed"}).json()
    assert body["category"] == "Hardware"
    assert body["confidence"] == 0.4
    assert body["needs_review"] is True
    assert body["review_reason"] == "low_confidence"


def test_confidence_at_the_threshold_does_not_need_review(client, monkeypatch, vectoriser):
    fake = FakeModel([0.50, 0.30, 0.20])            # exactly on the threshold: not below it
    monkeypatch.setattr(main, "artefacts", {"model": fake, "vectoriser": vectoriser})
    body = client.post("/predict", json={"text": "printer jammed"}).json()
    assert body["confidence"] == main.CONFIDENCE_THRESHOLD
    assert body["needs_review"] is False
