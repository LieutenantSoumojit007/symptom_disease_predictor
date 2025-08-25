"""
Symptom-to-Disease ML Predictor (toy demo)
-----------------------------------------
- Builds a synthetic dataset of symptoms -> disease labels
- Trains a RandomForest classifier
- Evaluates with accuracy, F1, ROC-AUC (macro)
- Saves model + metadata
- Supports CLI predictions and optional Streamlit UI

Usage:
1) Train/Evaluate (default):
   python symptom_disease_predictor.py

2) Predict from CLI (comma-separated symptoms):
   python symptom_disease_predictor.py --predict "fever, cough, sore throat"

3) Launch Streamlit app:
   streamlit run symptom_disease_predictor.py

Requirements:
- Python 3.9+
- pip install numpy pandas scikit-learn joblib streamlit (streamlit optional)
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

# ---------------------------
# 1) Domain setup (toy)
# ---------------------------

# Master symptom vocabulary (add/remove as needed)
SYMPTOMS: List[str] = [
    "fever", "chills", "cough", "sore throat", "runny nose", "sneezing",
    "headache", "nausea", "vomiting", "diarrhea", "fatigue", "body ache",
    "loss of taste", "loss of smell", "shortness of breath", "chest pain",
    "abdominal pain", "joint pain", "rash", "itching", "burning urination",
    "frequent urination", "back pain", "photophobia", "sensitivity to light",
    "dizziness", "dehydration", "bloating", "heartburn"
]

# Toy disease definitions with core symptoms (very simplified!)
DISEASE_PROFILES: Dict[str, List[str]] = {
    "Common Cold": ["cough", "runny nose", "sneezing", "sore throat", "headache"],
    "Influenza (Flu)": ["fever", "chills", "cough", "fatigue", "body ache", "headache"],
    "Migraine": ["headache", "nausea", "vomiting", "photophobia", "sensitivity to light"],
    "COVID-19": ["fever", "cough", "fatigue", "loss of taste", "loss of smell", "shortness of breath"],
    "Food Poisoning": ["nausea", "vomiting", "diarrhea", "abdominal pain", "dehydration"],
    "Gastroesophageal Reflux (GERD)": ["heartburn", "chest pain", "bloating"],
    "Urinary Tract Infection (UTI)": ["burning urination", "frequent urination", "abdominal pain", "back pain"],
    "Dengue": ["fever", "headache", "joint pain", "rash", "fatigue"],
    "Malaria": ["fever", "chills", "headache", "fatigue"],
}

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


@dataclass
class DatasetConfig:
    samples_per_disease: int = 350   # how many samples to synthesize per disease
    noise_symptoms_per_case: Tuple[int, int] = (0, 2)  # random extra symptoms
    missing_prob: float = 0.0        # probability to flip a present symptom to 0 (simulating missingness)
    negative_class_fraction: float = 0.20  # fraction of "none of the above" mixed cases
    test_size: float = 0.2


def synthesize_dataset(config: DatasetConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a toy dataset with multi-hot symptom vectors and a single disease label.
    """
    rows = []
    labels = []

    diseases = list(DISEASE_PROFILES.keys())

    # Positive samples per disease
    for disease, core_syms in DISEASE_PROFILES.items():
        for _ in range(config.samples_per_disease):
            symptoms = set(core_syms)

            # Add 0-2 random noise symptoms
            k = np.random.randint(config.noise_symptoms_per_case[0],
                                  config.noise_symptoms_per_case[1] + 1)
            noise_candidates = list(set(SYMPTOMS) - set(core_syms))
            if noise_candidates and k > 0:
                noise = np.random.choice(noise_candidates, size=k, replace=False)
                symptoms.update(noise)

            vector = [1 if s in symptoms else 0 for s in SYMPTOMS]

            # Simulate missingness (flip some 1s to 0)
            if config.missing_prob > 0:
                for i in range(len(vector)):
                    if vector[i] == 1 and np.random.rand() < config.missing_prob:
                        vector[i] = 0

            rows.append(vector)
            labels.append(disease)

    # Some negatives / ambiguous (random mixes that don't clearly match profiles)
    neg_samples = int(config.samples_per_disease * len(diseases) * config.negative_class_fraction)
    for _ in range(neg_samples):
        # choose random 1-4 symptoms
        k = np.random.randint(1, 5)
        chosen = set(np.random.choice(SYMPTOMS, size=k, replace=False))
        vector = [1 if s in chosen else 0 for s in SYMPTOMS]
        rows.append(vector)
        labels.append("Uncertain/Other")

    X = pd.DataFrame(rows, columns=SYMPTOMS)
    y = pd.Series(labels, name="disease")
    return X, y


def build_pipeline() -> Pipeline:
    """
    For binary features, we can impute and (optionally) scale.
    RandomForest is robust to unscaled binaries, but we keep a clean pipeline.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # scaler is not strictly necessary; kept for template structure
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", model)
    ])
    return pipe


def train_and_evaluate(config: DatasetConfig):
    print("🔧 Synthesizing dataset...")
    X, y = synthesize_dataset(config)

    print(f"Total samples: {len(X)} | Features (symptoms): {X.shape[1]} | Classes: {y.nunique()}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=RANDOM_SEED, stratify=y
    )

    print("🧠 Training model...")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    print("📊 Evaluating...")
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    # For ROC-AUC (macro), we need probability estimates and one-vs-rest binarization
    y_proba = None
    auc = None
    try:
        y_proba = pipe.predict_proba(X_test)
        # compute macro-roc-auc manually
        # map classes to indices
        classes = list(pipe.named_steps["clf"].classes_)
        y_true_bin = np.zeros((len(y_test), len(classes)))
        for i, label in enumerate(y_test):
            y_true_bin[i, classes.index(label)] = 1
        auc = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
    except Exception:
        pass

    print(f"\nAccuracy: {acc:.3f}")
    print(f"Macro F1: {f1:.3f}")
    if auc is not None:
        print(f"Macro ROC-AUC: {auc:.3f}")

    print("\nDetailed report:")
    print(classification_report(y_test, y_pred))

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/symptom_disease_model.joblib"
    meta_path = "artifacts/metadata.json"
    dump(pipe, model_path)
    metadata = {
        "symptoms": SYMPTOMS,
        "diseases": sorted(y.unique().tolist()),
        "random_seed": RANDOM_SEED,
        "config": {
            "samples_per_disease": config.samples_per_disease,
            "noise_symptoms_per_case": config.noise_symptoms_per_case,
            "missing_prob": config.missing_prob,
            "negative_class_fraction": config.negative_class_fraction,
            "test_size": config.test_size,
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n💾 Saved model -> {model_path}")
    print(f"💾 Saved metadata -> {meta_path}")
    return pipe


def load_artifacts(model_path="artifacts/symptom_disease_model.joblib",
                   meta_path="artifacts/metadata.json"):
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(
            "Model or metadata not found. Run training first: python symptom_disease_predictor.py"
        )
    pipe = load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return pipe, meta


def symptoms_to_vector(user_symptoms: List[str], vocab: List[str]) -> np.ndarray:
    user_symptoms = [s.strip().lower() for s in user_symptoms if s.strip()]
    return np.array([[1 if s in user_symptoms else 0 for s in vocab]])


def predict_from_text(input_text: str, top_k: int = 3):
    pipe, meta = load_artifacts()
    vocab = meta["symptoms"]
    classes = pipe.named_steps["clf"].classes_

    user_syms = [t.strip().lower() for t in input_text.split(",")]
    x = symptoms_to_vector(user_syms, vocab)

    proba = None
    try:
        proba = pipe.predict_proba(x)[0]
    except Exception:
        # Fallback to hard prediction
        label = pipe.predict(x)[0]
        return [{"disease": label, "prob": None, "rank": 1}]

    # Build ranking
    ranking = sorted(
        [{"disease": classes[i], "prob": float(proba[i])} for i in range(len(classes))],
        key=lambda d: d["prob"],
        reverse=True
    )[:max(1, top_k)]
    for i, r in enumerate(ranking, 1):
        r["rank"] = i
    return ranking


def print_prediction_table(ranking: List[Dict], user_input: str):
    print("\n🧾 Prediction (top candidates)")
    print("--------------------------------")
    print(f"Input symptoms: {user_input}")
    print("--------------------------------")
    for r in ranking:
        prob_str = f"{r['prob']:.3f}" if r["prob"] is not None else "n/a"
        print(f"{r['rank']}. {r['disease']:30s}  prob={prob_str}")
    print("\n⚠️ This is NOT a medical diagnosis. Consult a medical professional.")


# ---------------------------
# Optional: Streamlit UI
# ---------------------------
def run_streamlit():
    try:
        import streamlit as st
    except ImportError:
        print("Install streamlit first: pip install streamlit")
        return

    st.set_page_config(page_title="Symptom → Disease (Toy ML)", page_icon="🩺", layout="centered")
    st.title("🩺 Symptom → Disease Predictor (Toy Demo)")
    st.write(
        "Educational demo using a synthetic dataset. Not for medical use."
    )

    # Ensure artifacts exist
    if not (os.path.exists("artifacts/symptom_disease_model.joblib") and
            os.path.exists("artifacts/metadata.json")):
        st.warning("Model not found. Click the button below to train it.")
        if st.button("Train model now"):
            with st.spinner("Training..."):
                train_and_evaluate(DatasetConfig())
            st.success("Training complete. Reload the app.")
        st.stop()

    pipe, meta = load_artifacts()
    vocab = meta["symptoms"]
    classes = pipe.named_steps["clf"].classes_

    st.subheader("Select your symptoms")
    chosen = st.multiselect("Pick any that apply:", options=vocab)

    if st.button("Predict"):
        if not chosen:
            st.info("Please select at least one symptom.")
        else:
            x = symptoms_to_vector(chosen, vocab)
            try:
                proba = pipe.predict_proba(x)[0]
                df = pd.DataFrame({
                    "Disease": classes,
                    "Probability": proba
                }).sort_values("Probability", ascending=False).reset_index(drop=True)
            except Exception:
                pred = pipe.predict(x)[0]
                df = pd.DataFrame({"Disease": [pred], "Probability": [np.nan]})

            st.write("### Top results")
            st.dataframe(df.head(5), use_container_width=True)

            st.caption("⚠️ Not a diagnosis. Seek professional medical advice.")

    with st.expander("Model info"):
        st.json(meta)


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", type=str, default=None,
                        help="Comma-separated symptoms, e.g. 'fever, cough, sore throat'")
    parser.add_argument("--topk", type=int, default=3, help="Top-K predictions to show")
    parser.add_argument("--train_only", action="store_true", help="Only train/evaluate and exit")
    parser.add_argument("--streamlit", action="store_true", help="Run Streamlit UI (or use: streamlit run script.py)")
    args = parser.parse_args()

    if args.streamlit:
        run_streamlit()
        return

    # Train/evaluate if artifacts missing or train_only requested
    need_train = args.train_only or not (
        os.path.exists("artifacts/symptom_disease_model.joblib") and
        os.path.exists("artifacts/metadata.json")
    )

    if need_train:
        config = DatasetConfig()
        train_and_evaluate(config)
        if args.train_only and args.predict is None:
            return

    if args.predict:
        ranking = predict_from_text(args.predict, top_k=args.topk)
        print_prediction_table(ranking, args.predict)


if __name__ == "__main__":
    main()
