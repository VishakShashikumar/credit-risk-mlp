"""Train the Module 8 model and write the files the static Space serves.

Run:  python export_model.py
Writes model.json (weights and test metrics) and tests/fixtures/sample_rows.json
(200 test applicants with scikit-learn's probabilities, used to check predict.js).
"""
import json
from pathlib import Path

import numpy as np

import model

ROOT = Path(__file__).parent


def main():
    pipe, info, X_test, y_test = model.train()
    weights = model.export_weights(pipe, info)

    proba = pipe.predict_proba(X_test)[:, 1]
    gap = float(np.max(np.abs(model.forward(weights, X_test) - proba)))
    assert gap < 1e-9, f"exported weights disagree with scikit-learn by {gap}"

    (ROOT / "model.json").write_text(json.dumps(weights))
    sample = X_test.head(200)
    fixtures = {"rows": sample.to_dict(orient="records"),
                "sklearn_proba": pipe.predict_proba(sample)[:, 1].tolist()}
    out = ROOT / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sample_rows.json").write_text(json.dumps(fixtures))
    print(f"model.json written; max gap vs scikit-learn = {gap:.2e}")
    print(json.dumps(info["at_threshold"]))


if __name__ == "__main__":
    main()
