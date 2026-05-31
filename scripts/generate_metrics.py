import os
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.keras

from tensorflow import keras

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# MLflow Setup
# =====================================================

mlflow.set_tracking_uri("file:./mlruns")

# =====================================================
# Paths
# =====================================================

MODELS_DIR = "models"
DATA_DIR = "data"

# =====================================================
# Load Data
# =====================================================

X_test = np.load(
    os.path.join(DATA_DIR, "X_test_aug.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR, "y_test_aug.npy")
)

# =====================================================
# Feature Extraction for LR/RF
# =====================================================

def extract_features(X):

    mean = np.mean(X, axis=(1, 2))
    std = np.std(X, axis=(1, 2))
    max_val = np.max(X, axis=(1, 2))
    min_val = np.min(X, axis=(1, 2))
    energy = np.sum(X ** 2, axis=(1, 2))

    return np.column_stack([
        mean,
        std,
        max_val,
        min_val,
        energy
    ])

X_test_features = extract_features(X_test)

# =====================================================
# Load Models
# =====================================================

print("\nLoading Models...\n")

lr = joblib.load(
    os.path.join(
        MODELS_DIR,
        "logistic_regression.pkl"
    )
)

rf = joblib.load(
    os.path.join(
        MODELS_DIR,
        "random_forest.pkl"
    )
)

lstm = keras.models.load_model(
    os.path.join(
        MODELS_DIR,
        "lstm_model.h5"
    ),
    compile=False
)

bilstm = keras.models.load_model(
    os.path.join(
        MODELS_DIR,
        "bilstm_new.h5"
    ),
    compile=False
)

print("✓ All Models Loaded")

# =====================================================
# Predictions
# =====================================================

results = {}

# Logistic Regression
lr_probs = lr.predict_proba(X_test_features)[:, 1]
lr_preds = (lr_probs > 0.5).astype(int)

results["Logistic Regression"] = (
    lr_preds,
    lr_probs,
    lr
)

# Random Forest
rf_probs = rf.predict_proba(X_test_features)[:, 1]
rf_preds = (rf_probs > 0.5).astype(int)

results["Random Forest"] = (
    rf_preds,
    rf_probs,
    rf
)

# LSTM
lstm_probs = lstm.predict(X_test).flatten()
lstm_preds = (lstm_probs > 0.5).astype(int)

results["LSTM"] = (
    lstm_preds,
    lstm_probs,
    lstm
)

# BiLSTM
bilstm_probs = bilstm.predict(X_test).flatten()
bilstm_preds = (bilstm_probs > 0.5).astype(int)

results["BiLSTM"] = (
    bilstm_preds,
    bilstm_probs,
    bilstm
)

# =====================================================
# Artifacts Folder
# =====================================================

os.makedirs("artifacts", exist_ok=True)

# =====================================================
# Evaluation + MLflow Logging
# =====================================================

print("\nEvaluating Models...\n")

for model_name, (preds, probs, model_obj) in results.items():

    print(f"\n{'='*60}")
    print(model_name)
    print(f"{'='*60}")

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"AUC       : {auc:.4f}")

    # ==========================================
    # Confusion Matrix
    # ==========================================

    cm = confusion_matrix(y_test, preds)

    cm_path = f"artifacts/{model_name}_cm.png"

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.savefig(cm_path)
    plt.close()

    # ==========================================
    # ROC Curve
    # ==========================================

    fpr, tpr, _ = roc_curve(
        y_test,
        probs
    )

    roc_path = f"artifacts/{model_name}_roc.png"

    plt.figure(figsize=(6,5))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC={auc:.4f}"
    )

    plt.plot(
        [0,1],
        [0,1],
        '--'
    )

    plt.legend()

    plt.title(
        f"{model_name} ROC Curve"
    )

    plt.savefig(roc_path)
    plt.close()

    # ==========================================
    # Precision Recall Curve
    # ==========================================

    precision_vals, recall_vals, _ = precision_recall_curve(
        y_test,
        probs
    )

    pr_path = f"artifacts/{model_name}_pr.png"

    plt.figure(figsize=(6,5))

    plt.plot(
        recall_vals,
        precision_vals
    )

    plt.title(
        f"{model_name} Precision Recall Curve"
    )

    plt.savefig(pr_path)
    plt.close()

    # ==========================================
    # MLflow Logging
    # ==========================================

    mlflow.set_experiment(
        model_name
    )

    with mlflow.start_run(
        run_name="evaluation"
    ):

        mlflow.log_metric(
            "accuracy",
            acc
        )

        mlflow.log_metric(
            "precision",
            prec
        )

        mlflow.log_metric(
            "recall",
            rec
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            auc
        )

        mlflow.log_artifact(
            cm_path
        )

        mlflow.log_artifact(
            roc_path
        )

        mlflow.log_artifact(
            pr_path
        )

        if model_name in [
            "Logistic Regression",
            "Random Forest"
        ]:

            mlflow.sklearn.log_model(
                model_obj,
                artifact_path="model"
            )

        else:

            mlflow.keras.log_model(
                model_obj,
                artifact_path="model"
            )

        print(
            f"✓ Logged {model_name} to MLflow"
        )

print("\n")
print("="*70)
print("ALL MODELS LOGGED TO MLFLOW")
print("="*70)

