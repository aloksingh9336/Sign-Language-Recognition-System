# ============================================================
# TRAIN_FINAL_MODEL.PY
# SIGN LANGUAGE RECOGNITION SYSTEM
# PHASE 12 - FINAL MODEL TRAINING & OPTIMIZATION
# ============================================================

import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "landmark_data.csv"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "sign_language_model.pkl"
)

LABELS_PATH = os.path.join(
    MODELS_DIR,
    "class_labels.json"
)

METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "model_metrics.json"
)

REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

CONFUSION_MATRIX_PATH = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)


# ============================================================
# SETTINGS
# ============================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42

CV_FOLDS = 5

N_ITER_SEARCH = 20


# ============================================================
# HEADER
# ============================================================

print("=" * 70)

print(
    "SIGN LANGUAGE RECOGNITION SYSTEM"
)

print(
    "FINAL MODEL TRAINING & OPTIMIZATION"
)

print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

print()

print(
    "Loading dataset..."
)

if not os.path.exists(
    DATASET_PATH
):

    print()

    print(
        "ERROR: Dataset not found."
    )

    print(
        "Expected path:"
    )

    print(
        DATASET_PATH
    )

    print()

    print(
        "Please update DATASET_PATH"
    )

    print(
        "to match your actual dataset."
    )

    exit()


try:

    df = pd.read_csv(
        DATASET_PATH
    )

except Exception as e:

    print(
        "ERROR loading dataset:"
    )

    print(
        e
    )

    exit()


print()

print(
    "Dataset loaded successfully."
)

print(
    "Dataset shape:",
    df.shape
)


# ============================================================
# DISPLAY DATASET INFORMATION
# ============================================================

print()

print(
    "Dataset columns:"
)

print(
    list(df.columns)
)


# ============================================================
# FIND LABEL COLUMN
# ============================================================

possible_label_columns = [

    "label",

    "class",

    "target",

    "gesture",

    "sign"

]


label_column = None


for column in possible_label_columns:

    if column in df.columns:

        label_column = column

        break


if label_column is None:

    print()

    print(
        "ERROR: Could not find label column."
    )

    print(
        "Expected one of:"
    )

    print(
        possible_label_columns
    )

    exit()


print()

print(
    "Label column:",
    label_column
)


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

df = df.dropna()

print()

print(
    "Dataset after removing missing values:"
)

print(
    df.shape
)


# ============================================================
# SEPARATE FEATURES AND LABELS
# ============================================================

X = df.drop(
    columns=[
        label_column
    ]
)

y = df[
    label_column
]


# ============================================================
# REMOVE NON-NUMERIC COLUMNS
# ============================================================

X = X.select_dtypes(
    include=[
        np.number
    ]
)


print()

print(
    "Number of features:",
    X.shape[1]
)

print(
    "Number of samples:",
    X.shape[0]
)


# ============================================================
# CHECK FEATURE COUNT
# ============================================================

if X.shape[1] != 63:

    print()

    print(
        "WARNING:"
    )

    print(
        "Expected 63 numerical features."
    )

    print(
        "Found:",
        X.shape[1]
    )

    print(
        "Make sure the dataset matches"
    )

    print(
        "the MediaPipe landmark format."
    )


# ============================================================
# CLASS INFORMATION
# ============================================================

class_labels = sorted(
    y.astype(str).unique().tolist()
)

num_classes = len(
    class_labels
)


print()

print(
    "Number of classes:",
    num_classes
)

print()

print(
    "Classes:"
)

for index, label in enumerate(
    class_labels,
    start=1
):

    print(
        f"{index}. {label}"
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()

print(
    "Class distribution:"
)

print(
    y.value_counts()
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print()

print(
    "Creating stratified train/test split..."
)


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y

)


print()

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# BASELINE RANDOM FOREST
# ============================================================

print()

print(
    "=" * 70
)

print(
    "TRAINING BASELINE RANDOM FOREST"
)

print(
    "=" * 70
)


baseline_model = RandomForestClassifier(

    n_estimators=100,

    random_state=RANDOM_STATE,

    n_jobs=-1

)


baseline_model.fit(

    X_train,

    y_train

)


baseline_predictions = baseline_model.predict(
    X_test
)


baseline_accuracy = accuracy_score(

    y_test,

    baseline_predictions

)


print()

print(
    "Baseline Accuracy:",
    f"{baseline_accuracy * 100:.2f}%"
)


# ============================================================
# HYPERPARAMETER SEARCH
# ============================================================

print()

print(
    "=" * 70
)

print(
    "OPTIMIZING RANDOM FOREST"
)

print(
    "=" * 70
)


parameter_grid = {

    "n_estimators": [

        100,

        200,

        300,

        500

    ],

    "max_depth": [

        None,

        10,

        20,

        30,

        40

    ],

    "min_samples_split": [

        2,

        5,

        10

    ],

    "min_samples_leaf": [

        1,

        2,

        4

    ],

    "max_features": [

        "sqrt",

        "log2",

        None

    ]

}


cv_strategy = StratifiedKFold(

    n_splits=CV_FOLDS,

    shuffle=True,

    random_state=RANDOM_STATE

)


random_search = RandomizedSearchCV(

    estimator=RandomForestClassifier(

        random_state=RANDOM_STATE,

        n_jobs=-1

    ),

    param_distributions=parameter_grid,

    n_iter=N_ITER_SEARCH,

    cv=cv_strategy,

    scoring="accuracy",

    random_state=RANDOM_STATE,

    n_jobs=-1,

    verbose=1

)


print()

print(
    "Searching for best parameters..."
)

print(
    "This may take some time."
)


random_search.fit(

    X_train,

    y_train

)


best_model = random_search.best_estimator_


print()

print(
    "Best parameters:"
)

print(
    random_search.best_params_
)


print()

print(
    "Best cross-validation score:",
    f"{random_search.best_score_ * 100:.2f}%"
)


# ============================================================
# CROSS VALIDATION
# ============================================================

print()

print(
    "=" * 70
)

print(
    "CROSS-VALIDATION"
)

print(
    "=" * 70
)


cv_scores = cross_val_score(

    best_model,

    X_train,

    y_train,

    cv=cv_strategy,

    scoring="accuracy",

    n_jobs=-1

)


print()

print(
    "Cross-validation scores:"
)

for index, score in enumerate(

    cv_scores,

    start=1

):

    print(

        f"Fold {index}: "
        f"{score * 100:.2f}%"

    )


cv_mean = cv_scores.mean()

cv_std = cv_scores.std()


print()

print(
    "Mean CV Accuracy:",
    f"{cv_mean * 100:.2f}%"
)

print(
    "CV Standard Deviation:",
    f"{cv_std * 100:.2f}%"
)


# ============================================================
# FINAL TRAINING
# ============================================================

print()

print(
    "=" * 70
)

print(
    "TRAINING FINAL MODEL"
)

print(
    "=" * 70
)


best_model.fit(

    X_train,

    y_train

)


# ============================================================
# FINAL PREDICTION
# ============================================================

y_pred = best_model.predict(

    X_test

)


# ============================================================
# MODEL METRICS
# ============================================================

accuracy = accuracy_score(

    y_test,

    y_pred

)


precision = precision_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0

)


recall = recall_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0

)


f1 = f1_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0

)


print()

print(
    "=" * 70
)

print(
    "FINAL MODEL PERFORMANCE"
)

print(
    "=" * 70
)


print()

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1-Score  : {f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_test,

    y_pred,

    zero_division=0

)


print()

print(
    "Classification Report:"
)

print(

    report

)


with open(

    REPORT_PATH,

    "w",

    encoding="utf-8"

) as file:

    file.write(

        report

    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()

print(
    "Creating confusion matrix..."
)


cm = confusion_matrix(

    y_test,

    y_pred,

    labels=class_labels

)


fig, ax = plt.subplots(

    figsize=(14, 12)

)


display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=class_labels

)


display.plot(

    ax=ax,

    xticks_rotation=90,

    cmap="Blues",

    colorbar=True

)


plt.title(

    "Sign Language Recognition - Confusion Matrix"

)


plt.tight_layout()


plt.savefig(

    CONFUSION_MATRIX_PATH,

    dpi=300

)


plt.close()


print(

    "Confusion matrix saved."

)


# ============================================================
# SAVE MODEL
# ============================================================

print()

print(
    "Saving final model..."
)


joblib.dump(

    best_model,

    MODEL_PATH

)


print(

    "Model saved to:"

)

print(

    MODEL_PATH

)


# ============================================================
# SAVE CLASS LABELS
# ============================================================

with open(

    LABELS_PATH,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        class_labels,

        file,

        indent=4

    )


print()

print(

    "Class labels saved to:"

)

print(

    LABELS_PATH

)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "accuracy": round(

        float(accuracy),

        4

    ),

    "precision": round(

        float(precision),

        4

    ),

    "recall": round(

        float(recall),

        4

    ),

    "f1_score": round(

        float(f1),

        4

    ),

    "baseline_accuracy": round(

        float(baseline_accuracy),

        4

    ),

    "cross_validation_mean": round(

        float(cv_mean),

        4

    ),

    "cross_validation_std": round(

        float(cv_std),

        4

    ),

    "number_of_classes": num_classes,

    "number_of_features": X.shape[1],

    "training_samples": len(X_train),

    "testing_samples": len(X_test),

    "best_parameters": random_search.best_params_

}


with open(

    METRICS_PATH,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        metrics,

        file,

        indent=4

    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()

print(
    "=" * 70
)

print(
    "TRAINING COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)

print()

print(
    "FINAL RESULTS"
)

print(
    "-" * 70
)

print(

    f"Accuracy       : "
    f"{accuracy * 100:.2f}%"

)

print(

    f"Precision      : "
    f"{precision * 100:.2f}%"

)

print(

    f"Recall         : "
    f"{recall * 100:.2f}%"

)

print(

    f"F1-Score       : "
    f"{f1 * 100:.2f}%"

)

print(

    f"CV Mean        : "
    f"{cv_mean * 100:.2f}%"

)

print(

    f"Number Classes : "
    f"{num_classes}"

)

print(

    f"Features       : "
    f"{X.shape[1]}"

)

print()

print(
    "Saved Files:"
)

print(
    "1.",
    MODEL_PATH
)

print(
    "2.",
    LABELS_PATH
)

print(
    "3.",
    METRICS_PATH
)

print(
    "4.",
    REPORT_PATH
)

print(
    "5.",
    CONFUSION_MATRIX_PATH
)

print()

print(
    "=" * 70
)