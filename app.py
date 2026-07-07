import streamlit as st
import numpy as np
import pickle

# ── load the trained model bundle we saved earlier ────────────────────────────
@st.cache_resource
def load_model():
    with open("loan_model.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_model()
forest = bundle["forest"]
X_min = bundle["X_train_min"]
X_max = bundle["X_train_max"]

# ── prediction functions (same as our notebook) ───────────────────────────────
def predict_proba_one(tree, x):
    if tree["leaf"]:
        return tree["proba"]
    if x[tree["feature"]] <= tree["threshold"]:
        return predict_proba_one(tree["left"], x)
    else:
        return predict_proba_one(tree["right"], x)

def predict_forest_one(forest, x):
    votes = [predict_proba_one(tree, x) for tree in forest]
    return np.mean(votes)

# ── the web page ──────────────────────────────────────────────────────────────
st.title("Loan Default Risk Predictor")
st.write("Built from scratch with NumPy — logistic regression, decision tree, and random forest. This app uses the random forest (AUC 0.86).")

st.header("Enter Borrower Details")

col1, col2 = st.columns(2)

with col1:
    revolving = st.number_input("Credit utilization (0-1)", 0.0, 2.0, 0.3)
    age = st.number_input("Age", 18, 100, 45)
    late_30_59 = st.number_input("Times 30-59 days late", 0, 20, 0)
    debt_ratio = st.number_input("Debt ratio", 0.0, 5.0, 0.3)
    income = st.number_input("Monthly income", 0, 50000, 5000)

with col2:
    open_credit = st.number_input("Open credit lines/loans", 0, 60, 8)
    late_90 = st.number_input("Times 90+ days late", 0, 20, 0)
    real_estate = st.number_input("Real estate loans", 0, 30, 1)
    late_60_89 = st.number_input("Times 60-89 days late", 0, 20, 0)
    dependents = st.number_input("Number of dependents", 0, 20, 0)

if st.button("Predict Default Risk", type="primary"):
    raw = np.array([revolving, age, late_30_59, debt_ratio, income,
                    open_credit, late_90, real_estate, late_60_89, dependents])
    scaled = (raw - X_min) / (X_max - X_min + 1e-8)
    prob = predict_forest_one(forest, scaled)

    st.header("Result")
    st.metric("Default Probability", f"{prob:.1%}")

    if prob >= 0.15:
        st.error(f"HIGH RISK — {prob:.1%} chance of default. Recommend closer review.")
    else:
        st.success(f"LOWER RISK — {prob:.1%} chance of default.")

    st.caption("Threshold 0.15 balances catching defaulters vs false alarms.")