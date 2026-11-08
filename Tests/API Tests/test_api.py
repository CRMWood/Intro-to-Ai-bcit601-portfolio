from fastapi.testclient import TestClient
from main import app


# ---------- /health ----------

def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ---------- /categories ----------

def test_categories_lists_the_model_categories(client, model):
    response = client.get("/categories")
    assert response.status_code == 200
    assert response.json()["categories"] == model.classes_.tolist()


# ---------- /predict: the promise for a normal request ----------

def test_predict_response_has_the_promised_shape(client):
    response = client.post("/predict", json={"text": "There is a pothole in my street"})
    body = response.json()
    assert response.status_code == 200
    assert isinstance(body["category"], str)
    assert isinstance(body["confidence"], float)
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["needs_review"], bool)


def test_predicted_category_is_one_of_the_listed_categories(client):
    categories = client.get("/categories").json()["categories"]
    response = client.post("/predict", json={"text": "A powerline has fallen"})
    assert response.json()["category"] in categories


@pytest.mark.parametrize("text, expected_category", [
    ("A watermain has burst", "Water"),
    ("A traffic light on the corner of Morhouse Ave and Gasson st is flashing orange", "Road"),
    ("A bin near the childrens playground at Avonhead Park has overflowed", "Waste"),
    ("Handrail for the ", "Waste"),
])
def test_predict_gives_the_expected_category(client, text, expected_category):
    response = client.post("/predict", json={"text": text})
    assert response.json()["category"] == expected_category


def test_a_clear_ticket_does_not_need_review(client):
    response = client.post("/predict", json={"text": "I am locked out and need a password reset"})
    assert response.json()["needs_review"] is False
    assert response.json()["review_reason"] is None


# ---------- Wrong method or wrong path ----------

def test_get_on_predict_is_not_allowed(client):
    assert client.get("/predict").status_code == 405


def test_unknown_path_is_not_found(client):
    assert client.get("/nothing").status_code == 404
