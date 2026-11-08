import pytest


# ---------- /health test ----------

def test_health_returns_ok(client, results_log):
    response = client.get("/health")
    results_log("health: normal request", response)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ---------- /categories test----------

def test_categories_lists_the_model_categories(client, model, results_log):
    response = client.get("/categories")
    results_log("categories: normal request", response)
    assert response.status_code == 200
    assert response.json()["categories"] == model.classes_.tolist()


# ---------- /predict: the promise for a normal request ----------

def test_predict_response_has_the_promised_shape(client, results_log):
    response = client.post("/predict", json={"text": "There is a pothole in my street"})
    results_log("predict: pothole", response)
    body = response.json()
    assert response.status_code == 200
    assert isinstance(body["category"], str)
    assert isinstance(body["confidence"], float)
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["needs_review"], bool)


def test_predicted_category_is_one_of_the_listed_categories(client, results_log):
    categories = client.get("/categories").json()["categories"]
    response = client.post("/predict", json={"text": "A powerline has fallen"})
    results_log("predict: powerline", response)
    assert response.json()["category"] in categories


@pytest.mark.parametrize("text, expected_category", [
    ("A watermain has burst", "water"),
    ("A traffic light on the corner of Morhouse Ave and Gasson st is flashing orange", "road"),
    ("A bin near the childrens playground at Avonhead Park has overflowed", "waste"),
    ("Handrail for the steps has broken", "safety"),
])
def test_predict_gives_the_expected_category(client, text, expected_category, results_log):
    response = client.post("/predict", json={"text": text})
    results_log(f"predict: expects {expected_category}", response)
    assert response.json()["category"] == expected_category


# ---------- Wrong method or wrong path ----------

def test_get_on_predict_is_not_allowed(client, results_log):
    response = client.get("/predict")
    results_log("predict: wrong method (GET)", response)
    assert client.get("/predict").status_code == 405


def test_unknown_path_is_not_found(client, results_log):
    response = client.get("/nothing")
    results_log("unknown path", response)
    assert client.get("/nothing").status_code == 404

