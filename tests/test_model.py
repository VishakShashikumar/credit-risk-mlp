"""Checks that must pass before GitHub Actions deploys to the Hugging Face Space."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import model  # noqa: E402

# Quality floor for deployment. Module 8 measured F1 = 0.548 and recall = 0.558
# at the 0.29 cutoff; a model that falls below these floors is not shipped.
MIN_F1 = 0.50
MIN_RECALL = 0.50


@pytest.fixture(scope="module")
def trained():
    return model.train()


@pytest.fixture(scope="module")
def committed():
    return json.loads((ROOT / "model.json").read_text())


def test_data_has_expected_shape_and_labels():
    X, y = model.load_data()
    assert X.shape == (30000, 23)
    assert list(X.columns) == model.FEATURES
    assert set(y.unique()) == {0, 1}
    assert abs(y.mean() - 0.2212) < 0.001


def test_cleaning_merges_undocumented_codes():
    X, _ = model.load_data()
    assert set(X["EDUCATION"].unique()) <= {1, 2, 3, 4}
    assert set(X["MARRIAGE"].unique()) <= {1, 2, 3}


def test_retrained_model_meets_quality_floor(trained):
    _, info, _, _ = trained
    m = info["at_threshold"]
    assert info["test_rows"] == 6000
    assert m["f1"] >= MIN_F1, f"F1 {m['f1']} is below the deployment floor of {MIN_F1}"
    assert m["recall"] >= MIN_RECALL, f"Recall {m['recall']} is below the floor of {MIN_RECALL}"


def test_threshold_catches_more_defaulters_than_default_cutoff(trained):
    _, info, _, _ = trained
    assert info["at_threshold"]["defaulters_caught"] > info["at_default_0_50"]["defaulters_caught"]


def test_export_reproduces_scikit_learn(trained):
    pipe, info, X_test, _ = trained
    weights = model.export_weights(pipe, info)
    gap = np.max(np.abs(model.forward(weights, X_test) - pipe.predict_proba(X_test)[:, 1]))
    assert gap < 1e-9


def test_committed_model_json_meets_quality_floor(trained, committed):
    """The weights the Space actually serves must clear the same floor."""
    _, _, X_test, y_test = trained
    assert committed["threshold"] == model.THRESHOLD
    assert committed["features"] == model.FEATURES
    pred = (model.forward(committed, X_test) >= committed["threshold"]).astype(int)
    assert f1_score(y_test, pred) >= MIN_F1
    assert recall_score(y_test, pred) >= MIN_RECALL


def test_committed_metrics_match_committed_weights(trained, committed):
    """The model card in the app must report what the served weights really do."""
    _, _, X_test, y_test = trained
    pred = (model.forward(committed, X_test) >= committed["threshold"]).astype(int)
    assert round(f1_score(y_test, pred), 3) == committed["info"]["at_threshold"]["f1"]
    assert round(recall_score(y_test, pred), 3) == committed["info"]["at_threshold"]["recall"]
