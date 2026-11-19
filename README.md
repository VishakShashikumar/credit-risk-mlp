---
title: Credit Risk MLP
emoji: 💳
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Credit card default risk check (AAI6610 project)
---

# Credit default risk check

A web app that estimates the probability that a credit card holder defaults on next
month's payment. It deploys the model selected in my Module 8 assignment for AAI6610
(Applied Machine Learning, Northeastern University): a multilayer perceptron with hidden
layers of 64 and 32 units, used with a decision threshold of 0.29.

- **Live app:** https://huggingface.co/spaces/Vvshockk/credit-risk-mlp
- **Source and CI/CD:** https://github.com/VishakShashikumar/credit-risk-mlp

## How it works

Hugging Face now requires a paid plan for Gradio and Docker Spaces, so this runs on a free
**static** Space and the model runs in the visitor's browser.

- `model.py` loads the data, applies the Module 2 cleaning rules and trains the
  scikit-learn pipeline on the same 80/20 split used since Module 3 (random_state=20).
- `export_model.py` writes the fitted pipeline's numbers (one-hot categories, scaler means
  and standard deviations, network weights and biases, and test metrics) to `model.json`.
- `predict.js` repeats the pipeline's arithmetic in JavaScript, and `index.html` is the form.

Results on the 6,000-row held-out test set at the 0.29 cutoff: accuracy 79.47%,
precision 0.538, recall 0.558, F1 0.548.

## CI/CD

Every push to `main` runs `.github/workflows/ci-cd.yml` on GitHub Actions:

1. **Test:** checks that no file exceeds Hugging Face's 10 MB limit, retrains the model,
   and runs `pytest`. The build fails if F1 or recall at the 0.29 cutoff drops below 0.50,
   if the exported weights stop matching scikit-learn, or if the metrics shown in the app
   do not match the served weights. A Node test then checks that `predict.js` reproduces
   scikit-learn's probabilities for 200 held-out applicants.
2. **Deploy:** only if every test passes, pushes the repository to the Space.

Pull requests run the tests but never deploy. To update the model, change `model.py`,
run `python export_model.py`, and push.

## Data

Yeh, I.-C. (2009). *Default of credit card clients* [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C55S3H (CC BY 4.0). Stored as
`data/credit_default.csv`.

This is a course project and not a lending tool.
