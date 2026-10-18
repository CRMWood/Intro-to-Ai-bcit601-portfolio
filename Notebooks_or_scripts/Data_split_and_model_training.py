import pandas as pd
from config import PROCESSED_DIR, MODEL_DIR, TEST_DIR
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder # transforms the columns suburb and priority into positive numbers (1 or 0) so that no negative numbers turn up, a requirement for MulinominalNB
from sklearn.compose import ColumnTransformer # seperates request_text (which uses TF-IDF) from suburb and priority(OneHotEncoder)
from sklearn.pipeline import Pipeline #links steps together so that data flows through in the correct order
from sklearn.naive_bayes import MultinomialNB #chosen because it suits small text datasets well
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv(PROCESSED_DIR / "BCIT601_starter_service_requests_dataset_clean.csv")

# Features and label
X = df[["request_text", "suburb", "priority"]]
y = df["category"]

# 80/20 split (stratify keeps category proportions similar in both sets,
# which matters with only 80 rows)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Preprocessing: TF-IDF on the text, one-hot on suburb and priority
preprocess = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(stop_words="english", ngram_range=(1, 2)), "request_text"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["suburb", "priority"]),
    ]
)

# Full pipeline: preprocessing + classifier
model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", MultinomialNB(alpha=0.5)),
    ]
)

# Train
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# Evaluate on the test 20%
with open(TEST_DIR / "model_creation_test_results.txt", "a") as f: #creates a file for the output
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
    log("")
    log("=== First 5 test rows ===")
    test_results = pd.DataFrame({
        "request_text": X_test["request_text"].values[:5],
        "true_label": y_test.values[:5],
        "predicted_label": y_pred[:5],
    })
    log(test_results.to_string(index=False))

# Save the whole pipeline for the API
joblib.dump(model, MODEL_DIR / "model-CRW0262_BCIT601_request_classifier.joblib")

#Pulling the trained vectoriser from the pipeline and saving it for use
trained_vectoriser = model.named_steps["preprocess"].named_transformers_["text"]
joblib.dump(trained_vectoriser, MODEL_DIR / "vectoriser-CRW0262_BCIT601_request_classifier.joblib")
print("Model and vectoriser saved to the 'Model' directory.")
print("Inital test results saved to the 'Test' directory.")