
import pandas as pd
from config import PROCESSED_DIR, MODEL_DIR, TEST_DIR
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline #links steps together so that data flows through in the correct order
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv(PROCESSED_DIR / "BCIT601_starter_service_requests_dataset_clean.csv")

# Features and label
X = df["request_text"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = Pipeline(
    steps=[
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

y_pred = model.predict(X_test)
# Evaluate on the test 20%
with open(TEST_DIR / "Logistic_Regression_Model" / "Logistic_regression_model_creation_test_results.txt", "a") as f: #creates a file for the output
    def log(*args):
        print(*args) #sends any arguments to the terminal
        print(*args, file=f) #saves the same arguements to a file

    
    log("=== Train/test split (80/20) ===")
    log("Training rows:", len(X_train))
    log("Testing rows:", len(X_test))
    log()
    log("=== Evaluation ===")
    log("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    log(classification_report(y_test, y_pred, zero_division=0),)
    log()
    log("=== First 5 test rows ===")
    test_results = pd.DataFrame({
        "request_text": X_test.values[:5],
        "true_label": y_test.values[:5],
        "predicted_label": y_pred[:5],
    })
    log(test_results.to_string(index=False))

# Save the whole pipeline for the API
joblib.dump(model, MODEL_DIR / "Logistic_Regression" / "request_text_model-CRW0262_BCIT601_request_classifier.joblib")

#Pulling the trained vectoriser from the pipeline and saving it for use
trained_vectoriser = model.named_steps["tfidf"]
joblib.dump(trained_vectoriser, MODEL_DIR / "Logistic_Regression" / "request_text_vectoriser-CRW0262_BCIT601_request_classifier.joblib")
print("Model and vectoriser saved to the 'Model' directory.")
print("Inital test results saved to the 'Test' directory.")