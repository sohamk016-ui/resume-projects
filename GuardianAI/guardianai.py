"""Train and run an email classifier with TF-IDF and logistic regression."""
import argparse
import csv
from pathlib import Path

from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

LABELS = ["legitimate", "phishing"]


def read_examples(path):
    examples = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"text", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires text and label columns")
        for row in reader:
            message = (row["text"] or "").strip()
            label = (row["label"] or "").strip().lower()
            if not message or label not in LABELS:
                raise ValueError(f"CSV line {reader.line_num}: supply text and a legitimate or phishing label")
            key = " ".join(message.casefold().split())
            if key in examples and examples[key][1] != label:
                raise ValueError(f"CSV line {reader.line_num}: duplicate message has conflicting labels")
            examples[key] = (message, label)
    texts = [message for message, _ in examples.values()]
    labels = [label for _, label in examples.values()]
    if any(labels.count(label) < 4 for label in LABELS):
        raise ValueError("at least four unique examples per label are required")
    return texts, labels


def train(dataset, model_path):
    messages, labels = read_examples(dataset)
    train_text, test_text, train_labels, test_labels = train_test_split(
        messages, labels, test_size=0.25, stratify=labels, random_state=42
    )
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    model.fit(train_text, train_labels)
    predicted = model.predict(test_text)
    print(f"Unique examples: {len(messages)} ({len(train_text)} train, {len(test_text)} held out)")
    print(f"Held-out accuracy: {accuracy_score(test_labels, predicted):.1%}")
    print(classification_report(test_labels, predicted, labels=LABELS, zero_division=0))
    print(f"Confusion matrix (rows actual, columns predicted: {', '.join(LABELS)}):")
    print(confusion_matrix(test_labels, predicted, labels=LABELS))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)
    print(f"Model saved to {model_path.resolve()}")


def predict(model_path, message):
    if not message.strip():
        raise ValueError("email text is empty")
    model = load(model_path)
    probabilities = model.predict_proba([message])[0]
    predicted = model.classes_[probabilities.argmax()]
    print(f"Prediction: {predicted}")
    for label, probability in zip(model.classes_, probabilities):
        print(f"{label}: {probability:.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    training = actions.add_parser("train", help="train and evaluate on a labeled CSV")
    training.add_argument("csv", type=Path)
    prediction = actions.add_parser("predict", help="classify a single message")
    inputs = prediction.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--file", type=Path)
    for action in (training, prediction):
        action.add_argument("--model", type=Path, default=Path("output/guardianai.joblib"))

    args = parser.parse_args()
    try:
        if args.action == "train":
            train(args.csv, args.model)
        else:
            message = args.text if args.file is None else args.file.read_text(encoding="utf-8-sig")
            predict(args.model, message)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
