"""Credit default model used by the Hugging Face Space.

This reproduces the model selected in the Module 8 assignment: the original
multilayer perceptron (hidden layers 64 and 32) trained on the UCI Default of
Credit Card Clients data with the same cleaning steps, the same 80/20 split
(random_state=20) and the decision threshold of 0.29 that was chosen from
out-of-fold F1 on the training set.
"""
from functools import lru_cache
from pathlib import Path
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "credit_default.csv"
SEED = 20
TEST_SIZE = 0.20
# Chosen in Module 8 by sweeping thresholds on 3-fold out-of-fold predictions
# from the training set only; the test set was not used to pick it.
THRESHOLD = 0.29

NOMINAL = ["SEX", "EDUCATION", "MARRIAGE"]
PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMTS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMTS = [f"PAY_AMT{i}" for i in range(1, 7)]
FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + PAY_STATUS + BILL_AMTS + PAY_AMTS
# Same column order as Module 8: continuous columns first, then the PAY_x status columns.
CONTINUOUS = [c for c in FEATURES if c not in NOMINAL + PAY_STATUS]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Merge the undocumented EDUCATION and MARRIAGE codes, as in Module 2."""
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def load_data(path: Path = DATA_PATH):
    df = pd.read_csv(path).rename(columns={"default payment next month": "default"})
    df = clean(df)
    return df[FEATURES], df["default"]


def build_pipeline(seed: int = SEED) -> Pipeline:
    prep = ColumnTransformer(
        [("nom", OneHotEncoder(handle_unknown="ignore"), NOMINAL),
         ("num", StandardScaler(), CONTINUOUS + PAY_STATUS)],
        remainder="drop",
    )
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                        alpha=1e-4, batch_size=64, learning_rate_init=1e-3, max_iter=500,
                        early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
                        random_state=seed)
    return Pipeline([("prep", prep), ("model", mlp)])


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred)), 3),
        "recall": round(float(recall_score(y_true, pred)), 3),
        "f1": round(float(f1_score(y_true, pred)), 3),
        "defaulters_caught": int(tp),
        "defaulters_missed": int(fn),
        "false_alarms": int(fp),
    }


@lru_cache(maxsize=1)
def train():
    """Fit the pipeline once per process and score it on the held-out test set."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED)
    start = time.time()
    pipe = build_pipeline().fit(X_train, y_train)
    seconds = time.time() - start
    proba = pipe.predict_proba(X_test)[:, 1]
    info = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_defaults": int(y_test.sum()),
        "epochs": int(pipe.named_steps["model"].n_iter_),
        "train_seconds": round(seconds, 1),
        "at_threshold": evaluate(y_test, proba, THRESHOLD),
        "at_default_0_50": evaluate(y_test, proba, 0.50),
    }
    return pipe, info, X_test, y_test


def score(pipe: Pipeline, applicant: dict) -> tuple[float, bool]:
    """Return (probability of default, flagged) for one applicant."""
    row = clean(pd.DataFrame([applicant], columns=FEATURES))
    prob = float(pipe.predict_proba(row)[0, 1])
    return prob, prob >= THRESHOLD


def export_weights(pipe: Pipeline, info: dict) -> dict:
    """Turn the fitted pipeline into plain numbers the browser can use.

    The static Space has no Python runtime, so the one-hot categories, the
    scaler's means and standard deviations, and the network's weights and
    biases are written to model.json and evaluated by predict.js.
    """
    prep = pipe.named_steps["prep"]
    onehot = prep.named_transformers_["nom"]
    scaler = prep.named_transformers_["num"]
    mlp = pipe.named_steps["model"]
    assert mlp.out_activation_ == "logistic" and mlp.activation == "relu"
    return {
        "threshold": THRESHOLD,
        "features": FEATURES,
        "nominal": {col: [int(c) for c in cats] for col, cats in zip(NOMINAL, onehot.categories_)},
        "scaled_columns": CONTINUOUS + PAY_STATUS,
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v) for v in scaler.scale_],
        "layers": [{"weights": W.tolist(), "bias": b.tolist()}
                   for W, b in zip(mlp.coefs_, mlp.intercepts_)],
        "info": info,
    }


def forward(weights: dict, X: pd.DataFrame):
    """NumPy version of predict.js, used to check the exported numbers."""
    import numpy as np
    X = clean(X)
    parts = [(X[col].to_numpy()[:, None] == np.array(cats)[None, :]).astype(float)
             for col, cats in weights["nominal"].items()]
    num = X[weights["scaled_columns"]].to_numpy(dtype=float)
    parts.append((num - np.array(weights["mean"])) / np.array(weights["scale"]))
    a = np.hstack(parts)
    layers = weights["layers"]
    for i, layer in enumerate(layers):
        a = a @ np.array(layer["weights"]) + np.array(layer["bias"])
        a = np.maximum(a, 0) if i < len(layers) - 1 else 1 / (1 + np.exp(-a))
    return a[:, 0]
