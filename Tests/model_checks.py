"""Three checks on the model, measured on the test set.

"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from conftest import BASE_DIR, MODEL_LOCATION, PROCESSED_DIR, OUTPUTS_DIR



CSV_FILE = "BCIT601_starter_service_requests_dataset_clean.csv"
TEXT_COLUMN = "request_text"
LABEL_COLUMN = "category"

model = joblib.load(MODEL_LOCATION / "request_text_model-CRW0262_BCIT601_request_classifier.joblib")
vectoriser = joblib.load(MODEL_LOCATION / "request_text_vectoriser-CRW0262_BCIT601_request_classifier.joblib")



data = pd.read_csv(PROCESSED_DIR / CSV_FILE)
_, test_texts, _, test_labels = train_test_split(
    data[TEXT_COLUMN], data[LABEL_COLUMN], test_size=0.2, random_state=42,
    stratify=data[LABEL_COLUMN],
)
test_texts = test_texts.tolist()
test_labels = test_labels.tolist()
total = len(test_texts)


def predict_all(texts):
    """Return the predicted categories and the confidence for every text."""
    features = vectoriser.transform(texts)
    probabilities = model.predict_proba(features)
    categories = [str(model.classes_[row.argmax()]) for row in probabilities]
    confidences = [float(row.max()) for row in probabilities]
    return categories, confidences


base_categories, base_confidences = predict_all(test_texts)

with open(OUTPUTS_DIR / "Check_Model_Tests" / "model_test_results.txt", "w") as f: #creates a file for the output NB will write over any previous results.
    def log(*args):
        print(*args) #sends any arguments to the terminal
        print(*args, file=f) #saves the same arguements to a file

    # ---------- Check 1: perturbation ----------
    #first set up some functions to help run the tests
    #adds puctution half way through the sentence, without breaking a word
    def extra_punct(text):
        #splits the words at spaces and finds the halfway point
        words = text.split()
        half = len(words) // 2
        #adds the comma
        words[half - 1] = words[half - 1] + ","
        #returns the sentence and the comma
        return " ".join(words)
        
    def create_typo(text):
        #adds a letter to the first word in a sentence that is longer than three letters
        words = text.split()
        for i, word in enumerate(words):
            if len(word) > 3:
                words[i] = word[:2] + "z" + word[2:]
        return " ".join(words)
                

    # Each entry is a name and a small function that changes one text.
    PERTURBATIONS = {
        #makes the text upper case
        "UPPER CASE": lambda text: text.upper(),

        #adds puctuation to the end
        "Extra punctuation": lambda text: extra_punct(text),

        #Adds greeting to the begining and a thank you at the end
        "Greeting and Thank you": lambda text: "Hi team, " + text + " Thanks.",
        
        #creates a typo in the request
        "Typo": lambda text : create_typo(text),

        #shortens a request to just the first three words
        "Just the first three words of the request": lambda text : " ".join(text.split()[:3])
    }

    log(f"CHECK 1: perturbation ({total} test texts)")
    log(f"{'Change made to the text':36} {'Changed answers':>15} {'Mean confidence':>16}")
    log(f"{'No change (baseline)':36} {'0 of ' + str(total):>15} {sum(base_confidences) / total:>16.3f}")
    for name, change in PERTURBATIONS.items():
        new_categories, new_confidences = predict_all([change(text) for text in test_texts])
        flips = 0
        #checks if the old prediction is the same as the new, if not it adds one to flips
        for before, after in zip(base_categories, new_categories):
            if before != after:
                flips += 1
        mean_confidence = sum(new_confidences) / len(new_confidences)
        log(f"{name:36} {str(flips) + ' of ' + str(total):>15} {mean_confidence:>16.3f}")


    # ---------- Check 2: confidence threshold ----------

    correct = [b == t for b, t in zip(base_categories, test_labels)]
    mistakes = correct.count(False)

    log(f"\nCHECK 2: confidence threshold ({mistakes} mistakes among {total} test texts)")
    log(f"{'Threshold':>9} {'Flagged':>9} {'Mistakes caught':>16} {'Correct flagged':>16}")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        flagged = 0
        caught = 0
        correct_flagged = 0
        for is_correct, confidence in zip(correct, base_confidences):
            if confidence < threshold:
                flagged += 1
                if is_correct:
                    correct_flagged += 1
                else:
                    caught += 1
                pass
        log(f"{threshold:>9.1f} {flagged:>9} {str(caught) + ' of ' + str(mistakes):>16} {correct_flagged:>16}")


    # ---------- Check 3: slices ----------

    log("\nCHECK 3: accuracy by text length")
    log(f"{'Slice':22} {'Texts':>8} {'Correct':>8} {'Accuracy':>9}")
    SLICES = {"up to 8 words": (1, 8), "9 to 10 words": (9, 10), "11 or more words": (11, 1000)}
    for name, (low, high) in SLICES.items():
        count = 0
        right = 0
        for text, is_correct in zip(test_texts, correct):
            word_count = len(text.split())
            if low <= word_count <= high:
                count += 1
                if is_correct:
                    right += 1
        accuracy = right / count if count > 0 else 0
        log(f"{name:22} {count:>8} {right:>8} {accuracy:>9.2f}")
