"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: ExploreAI Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit app: News Article Classifier (3-model demo)
# File: Streamlit/base_app.py

import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)

# -----------------------------
# UI CONFIG
# -----------------------------
st.set_page_config(page_title="News Classifier", page_icon="📰", layout="wide")
st.markdown("<h1>📰 News Classification App</h1>", unsafe_allow_html=True)
st.caption("Compare three classic text models and classify new articles.")

# -----------------------------
# Light text cleaner to keep behaviour close to your notebook
# (We keep digits; strip URLs/symbols; normalize spaces)
# -----------------------------
URL_RE = re.compile(r"http\S+|www\S+|@\w+|#\w+")
SYMBOL_RE = re.compile(r"[^a-z0-9\s]")
SPACE_RE = re.compile(r"\s+")

def preclean(s: str) -> str:
    s = s or ""
    t = s.lower()
    t = URL_RE.sub(" ", t)
    t = re.sub(r"n['’]t\b", " not", t)
    t = re.sub(r"['’]re\b", " are", t)
    t = re.sub(r"['’]ve\b", " have", t)
    t = SYMBO_RE.sub(" ", t) if False else SYMBOL_RE.sub(" ", t)  # keep readable
    t = re.sub(r"\b(not|no|never)\s+(\w+)", r"\1_\2", t)
    t = SPACE_RE.sub(" ", t).strip()
    return t

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data():
    train = pd.read_csv("Data/processed/train.csv")
    test = pd.read_csv("Data/processed/test.csv")
    for df in (train, test):
        for c in ["headlines", "content", "category"]:
            if c in df.columns:
                df[c] = df[c].fillna("")
    train["data"] = (train["headlines"] + " " + train["content"]).apply(preclean)
    test["data"]  = (test["headlines"]  + " " + test["content"]).apply(preclean)
    return train, test

train_df, test_df = load_data()

# -----------------------------
# Build 3 pipelines (cached)
# -----------------------------
@st.cache_resource(show_spinner=True)
def build_models(train_df, test_df, max_feats=5000, ngram=(1,2), test_size=0.2, seed=42):
    X = train_df["data"].values
    y = train_df["category"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    def pipe(est):
        return Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=lambda s: s, tokenizer=None,
                                      stop_words="english",
                                      max_features=max_feats,
                                      ngram_range=ngram)),
            ("clf", est),
        ])

    models = {
        "Logistic Regression": pipe(LogisticRegression(max_iter=2000)),
        "Linear SVM (LinearSVC)": pipe(LinearSVC()),
        "Multinomial Naive Bayes": pipe(MultinomialNB())
    }

    results = {}
    for name, p in models.items():
        p.fit(X_train, y_train)
        pred = p.predict(X_val)
        acc = accuracy_score(y_val, pred)
        f1m = f1_score(y_val, pred, average="macro")
        results[name] = {
            "pipeline": p,
            "val_acc": acc,
            "val_f1_macro": f1m,
        }

    # also compute a test set score for the best model (optional)
    best_name = max(results, key=lambda k: results[k]["val_f1_macro"])
    best_pipe = results[best_name]["pipeline"]
    y_test = test_df["category"].values
    X_test = test_df["data"].values
    pred_test = best_pipe.predict(X_test)
    results[best_name]["test_acc"] = accuracy_score(y_test, pred_test)
    results[best_name]["test_f1_macro"] = f1_score(y_test, pred_test, average="macro")

    return results, (X_train, X_val, y_train, y_val)

results, splits = build_models(train_df, test_df)

# -----------------------------
# Sidebar
# -----------------------------
page = st.sidebar.radio("Navigate", ["Overview", "EDA", "Compare Models", "Confusion Matrix", "Predict", "Team"])
st.sidebar.markdown("---")
st.sidebar.write("Max features: 5000 · ngrams: (1,2)")
st.sidebar.write(f"Train size: {len(train_df):,} · Test size: {len(test_df):,}")

# -----------------------------
# Pages
# -----------------------------
if page == "Overview":
    st.subheader("Project Overview")
    st.write(
        "This app trains three classic text classifiers on a news dataset and "
        "lets you compare performance and run predictions. "
        "Models are trained in cached pipelines so switching pages is snappy."
    )
    st.markdown("**Models included**")
    st.write("- Logistic Regression\n- Linear SVM (LinearSVC)\n- Multinomial Naive Bayes")
    best = max(results, key=lambda k: results[k]["val_f1_macro"])
    st.success(f"Current best (by macro-F1 on validation): **{best}**")

elif page == "EDA":
    st.subheader("Quick EDA")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Train class distribution**")
        st.bar_chart(train_df["category"].value_counts())
    with col2:
        st.write("**Test class distribution**")
        st.bar_chart(test_df["category"].value_counts())

    st.write("**Sample rows**")
    st.dataframe(train_df.sample(min(5, len(train_df)), random_state=7)[["headlines", "content", "category"]])

elif page == "Compare Models":
    st.subheader("Validation Metrics")
    table = (
        pd.DataFrame({
            "Model": list(results.keys()),
            "Val Accuracy": [round(results[k]["val_acc"], 4) for k in results],
            "Val Macro-F1": [round(results[k]["val_f1_macro"], 4) for k in results],
        })
        .sort_values("Val Macro-F1", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(table, use_container_width=True)

    st.caption("Higher macro-F1 means better balance across categories.")

    # Optional MLflow log toggle
    with st.expander("Log this comparison to MLflow (optional)"):
        if st.button("Log best model & metrics"):
            try:
                import mlflow, mlflow.sklearn
                best = table.iloc[0]["Model"]
                bp = results[best]["pipeline"]
                with mlflow.start_run(run_name=f"streamlit_{best.replace(' ', '_').lower()}"):
                    mlflow.log_param("tfidf_max_features", 5000)
                    mlflow.log_param("tfidf_ngram_range", "(1,2)")
                    mlflow.log_param("model", best)
                    mlflow.log_metric("val_acc", float(results[best]["val_acc"]))
                    mlflow.log_metric("val_f1_macro", float(results[best]["val_f1_macro"]))
                    if "test_acc" in results[best]:
                        mlflow.log_metric("test_acc", float(results[best]["test_acc"]))
                        mlflow.log_metric("test_f1_macro", float(results[best]["test_f1_macro"]))
                    mlflow.sklearn.log_model(bp, artifact_path="model")
                st.success("Logged to MLflow.")
            except Exception as e:
                st.error(f"MLflow logging failed: {e}")

elif page == "Confusion Matrix":
    st.subheader("Confusion Matrix (best model on validation set)")
    best_name = max(results, key=lambda k: results[k]["val_f1_macro"])
    best_pipe = results[best_name]["pipeline"]
    _, X_val, _, y_val = splits  # we stored (X_train, X_val, y_train, y_val)
    preds = best_pipe.predict(X_val)

    labels = sorted(pd.Series(y_val).unique())
    cm = confusion_matrix(y_val, preds, labels=labels)
    st.write(f"**Best model:** {best_name}")
    st.write(pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                          columns=[f"pred_{l}" for l in labels]))
    st.caption("Diagonal = correct. Off-diagonal = confusions.")

elif page == "Predict":
    st.subheader("Try a Prediction")
    model_name = st.selectbox("Choose a model", list(results.keys()))
    text = st.text_area("Paste a headline and/or article content:", height=180,
                        placeholder="e.g., Shares rise after policy decision as markets rally…")
    if st.button("Predict", use_container_width=True):
        if not text.strip():
            st.warning("Enter some text to classify.")
        else:
            pipe = results[model_name]["pipeline"]
            pred = pipe.predict([preclean(text)])[0]
            st.success(f"Predicted category: **{pred}**")

elif page == "Team":
    st.subheader("Team & Notes")
    st.write("- Add your team names and roles here.")
    st.write("- Outline what each page does and any assumptions.")
    st.write("- Link to your repo and deployment if hosted.")

# Footer
st.markdown("---")
st.caption("Tip: use the Compare page to pick your best model, then the Predict page for demos.")
