import pandas as pd
from config import CLEAN_DATA, MODEL_MODEL, MODEL_VECTORISER
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder # transforms the columns suburb and priority into positive numbers (1 or 0) so that no negative numbers turn up, a requirement for MulinominalNB
from sklearn.compose import ColumnTransformer # seperates request_text (which uses TF-IDF) from suburb and priority(OneHotEncoder)
from sklearn.pipeline import Pipeline #links steps together so that data flows through in the correct order
from sklearn.naive_bayes import MultinomialNB #chosen because it suits small text datasets well
from sklearn.metrics import accuracy_score, classification_report

vectoriser = TfidfVectorizer()


df = pd.read_csv(CLEAN_DATA)

# Features and label
X = df[["request_text", "suburb", "priority"]]
y = df["category"]

# 80/20 split (stratify keeps category proportions similar in both sets,
# which matters with only 80 rows)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=== Train/test split (80/20) ===")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print()

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

# Evaluate on the test 20%
y_pred = model.predict(X_test)
print("=== Evaluation ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print(classification_report(y_test, y_pred, zero_division=0))

# Save the whole pipeline for the API
joblib.dump(model, MODEL_MODEL)
joblib.dump(vectoriser, MODEL_VECTORISER)
print("Model saved to the 'Model' directory")
