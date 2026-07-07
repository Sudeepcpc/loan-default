# Loan Default Prediction — Built From Scratch (NumPy Only)

Predicting whether a loan applicant will default, using three machine learning models — **logistic regression, a decision tree, and a random forest — all implemented from scratch in NumPy**, with no scikit-learn, XGBoost, or any ML library doing the work.

**Live demo:** _add your Streamlit link here after deploying_
**Author:** Sudeep C P — [github.com/Sudeepcpc](https://github.com/Sudeepcpc)

---

## The Problem

Banks lose money when borrowers default (fail to repay). Before approving a loan, a lender wants to estimate: *how likely is this applicant to default?* Using only an applicant's information — income, age, debt, and payment history — this project builds models that learn from 150,000 past borrowers to score a new applicant's default risk. It is a **binary classification** problem on real, imbalanced financial data.

**Dataset:** [Give Me Some Credit](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset) — 150,000 borrowers, 10 features, target `SeriousDlqin2yrs`. Only **6.7%** of borrowers defaulted, making class imbalance a central challenge.

---

## Why From Scratch?

Anyone can call `LogisticRegression().fit()`. The goal here was to understand and implement the machinery underneath: the sigmoid, cross-entropy loss, gradient descent, Gini impurity, information gain, recursive tree building, and bootstrap aggregation — each written by hand and verified against theory. In production I would use scikit-learn or XGBoost; this project exists to prove I understand what those libraries do internally.

Libraries used: **NumPy** (array math), **pandas** (loading the CSV), **matplotlib** (plots), **Streamlit** (the demo app). No ML library.

---

## Results — Three-Model Comparison

All models evaluated on the same 30,000-row held-out test set. **AUC** (threshold-independent ranking quality) is the fair comparison metric.

| Model | AUC | Precision | Recall | F1 | Notes |
|---|---|---|---|---|---|
| Logistic Regression | 0.636 | 0.104 | 0.517 | 0.173 | Sensitive to feature scaling |
| Decision Tree | 0.837 | 0.544 | 0.107 | 0.179 | Robust to scale, high precision |
| **Random Forest** | **0.861** | **0.342** | **0.568** | **0.427** | **Best across every metric** |

The random forest is the winner — better ranking, fewer false alarms, and more defaulters caught than the other models.

---

## Key Findings (the interesting part)

**1. Accuracy is a misleading metric on imbalanced data.**
At the default 0.5 threshold, logistic regression scored **93% accuracy** — while catching **zero of 2,020 real defaulters**. It achieved high accuracy simply by predicting "won't default" for everyone. This is why the project relies on precision, recall, F1, and AUC instead of accuracy.

**2. Threshold tuning turned a "useless" model into a useful one.**
The same logistic regression, evaluated at a tuned threshold of 0.07, caught **1,045 of 2,020 defaulters (52% recall)**. The model was never broken — it was being measured and thresholded incorrectly. There is no single "correct" threshold; it depends on the relative cost of a missed defaulter versus a false alarm.

**3. Feature scaling explained the performance gap between models.**
Logistic regression's modest AUC (0.636) was traced to min-max scaling being distorted by extreme outliers (e.g. a $3M income compressing all other incomes toward zero), which muted the most predictive features. Tree-based models split on raw thresholds ("was the borrower ever 90+ days late?"), so they are immune to feature scale — which is why the decision tree and random forest jumped to 0.84 and 0.86. This is a concrete demonstration of why tree ensembles often dominate on messy tabular data.

**4. The strongest predictor was payment history, not income.**
Correlation analysis and the tree's first split both identified *past-due payment history* as the most predictive signal — far more than income or debt ratio. In plain terms: it is not how much money a borrower has, but how reliably they repay.

---

## Fairness & Bias Considerations

Credit models carry real ethical and legal risk, because they can encode and amplify discrimination. Before any real-world deployment, this model would require:

- **Protected-attribute auditing** — checking whether predictions differ systematically across gender, age, region, or other protected groups, even though those attributes are not direct inputs (proxies can leak through correlated features).
- **Disparate-impact analysis** — measuring whether approval/rejection rates differ unfairly across groups.
- **Explainability** — providing per-decision reasons, which regulations increasingly require for credit decisions.
- **Human oversight** — the model should support, not replace, a human decision-maker.

This project treats the model as a decision *aid*, not an automated approver. Including this analysis reflects that responsible ML is about more than accuracy.

---

## Project Structure

```
loan-default/
├── data/
│   └── cs-training.csv        # dataset (download from Kaggle)
├── analysis.ipynb             # full pipeline: EDA → 3 models → evaluation
├── app.py                     # Streamlit web app (random forest)
├── loan_model.pkl             # saved trained forest + scaling values
└── README.md
```

---

## What's Inside the Notebook

- **Data preparation** — loading, missing-value handling (median fill), EDA (histograms, correlation heatmap), an 80/20 train/test split, and min-max scaling fit on the training set only (no data leakage).
- **Logistic regression from scratch** — `sigmoid`, prediction, cross-entropy cost, gradients `(p − y)·x`, and a gradient-descent training loop. Loss converged smoothly from 0.69 to 0.24.
- **Evaluation from scratch** — confusion matrix, precision, recall, F1, a threshold sweep, and a hand-built ROC curve with trapezoidal AUC.
- **Decision tree from scratch** — Gini impurity, information gain, and recursive splitting with depth and min-sample stopping conditions.
- **Random forest from scratch** — bootstrap aggregation (bagging) plus random feature subsets per split, with probability outputs averaged across 15 trees.

---

## Running It Locally

```bash
# 1. clone the repo
git clone https://github.com/Sudeepcpc/loan-default.git
cd loan-default

# 2. create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. install dependencies
pip install numpy pandas matplotlib streamlit jupyter

# 4. (optional) explore the full pipeline
jupyter notebook analysis.ipynb

# 5. run the web app
streamlit run app.py
```

The app loads the pre-trained `loan_model.pkl`, so it runs instantly without retraining.

---

## Possible Improvements (v2)

- **Robust scaling / outlier clipping** to lift logistic regression's ceiling.
- **Cost-sensitive thresholds** driven by the bank's actual cost of a default vs. a false alarm.
- **A production rebuild** using scikit-learn / XGBoost pipelines, hyperparameter tuning, experiment tracking (MLflow), and drift monitoring — to contrast the from-scratch learning version with a deployable one.

---

## Tech Stack

Python · NumPy · pandas · matplotlib · Streamlit

Built as a from-scratch machine learning study — every algorithm implemented and explained line by line.
