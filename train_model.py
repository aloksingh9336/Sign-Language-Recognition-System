import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "landmark_data.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "sign_language_model.pkl"
)

LABEL_FILE = os.path.join(
    MODEL_DIR,
    "class_labels.json"
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SIGN LANGUAGE RECOGNITION - MODEL TRAINING")
print("=" * 70)

print("\nDataset file:")
print(DATASET_FILE)

print("\nModel will be saved to:")
print(MODEL_FILE)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_FILE):
    print("\nERROR: Dataset CSV not found!")
    print("Please run:")
    print("python prepare_dataset.py")
    exit()


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATASET_FILE)

print("\nDataset shape:")
print(df.shape)

print("\nDataset columns:")
print(list(df.columns))


# ============================================================
# FIND LABEL COLUMN
# ============================================================

possible_label_columns = [
    "label",
    "class",
    "target",
    "sign"
]

label_column = None

for column in possible_label_columns:
    if column in df.columns:
        label_column = column
        break

if label_column is None:
    print("\nERROR: Could not find label column.")
    print("Available columns:")
    print(list(df.columns))
    exit()


print("\nLabel column detected:")
print(label_column)


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

df = df.dropna()

print("\nDataset shape after removing empty rows:")
print(df.shape)


# ============================================================
# PREPARE FEATURES AND LABELS
# ============================================================

X = df.drop(columns=[label_column])
y = df[label_column]


# Convert feature values to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Remove rows containing invalid numeric values
valid_rows = ~X.isna().any(axis=1)

X = X[valid_rows]
y = y[valid_rows]


# Convert to NumPy arrays
X = X.values
y = y.values


print("\nNumber of samples:")
print(len(X))

print("\nNumber of features:")
print(X.shape[1])

print("\nNumber of classes:")
print(len(np.unique(y)))

print("\nClasses:")
print(sorted(np.unique(y).tolist()))


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

unique_classes, class_counts = np.unique(
    y,
    return_counts=True
)

for class_name, count in zip(
    unique_classes,
    class_counts
):
    print(f"{str(class_name):10s}: {count}")


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("SPLITTING DATASET")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:")
print(len(X_train))

print("Testing samples:")
print(len(X_test))


# ============================================================
# CREATE RANDOM FOREST MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST MODEL")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

print("\nTraining started...")

model.fit(
    X_train,
    y_train
)

print("Training completed successfully.")


# ============================================================
# MODEL PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING MODEL")
print("=" * 70)

y_pred = model.predict(X_test)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

labels = sorted(
    np.unique(y)
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\nClasses:")
print(labels)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

joblib.dump(
    model,
    MODEL_FILE
)

print("\nModel saved successfully:")
print(MODEL_FILE)


# ============================================================
# SAVE CLASS LABELS
# ============================================================

class_labels = [
    str(label)
    for label in model.classes_
]

with open(
    LABEL_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_labels,
        file,
        indent=4
    )


print("\nClass labels saved successfully:")
print(LABEL_FILE)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"\nTotal samples used : {len(X)}")
print(f"Training samples   : {len(X_train)}")
print(f"Testing samples    : {len(X_test)}")
print(f"Number of classes  : {len(class_labels)}")
print(f"Accuracy            : {accuracy * 100:.2f}%")

print("\nModel file:")
print(MODEL_FILE)

print("\nLabels file:")
print(LABEL_FILE)

print("\nNext step:")
print("Integrate the trained model with real-time webcam recognition.")

print("=" * 70)