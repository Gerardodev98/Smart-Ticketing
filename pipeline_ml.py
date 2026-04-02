"""
pipeline_ml.py
--------------
Pipeline ML completa per classificazione ticket:
  1. Carica il dataset sintetico
  2. Preprocessing testuale (minuscole, rimozione punteggiatura)
  3. Addestra due modelli Naive Bayes (categoria + priorità)
  4. Valuta su test set (accuracy, F1-macro, confusion matrix)
  5. Salva i modelli in models/
  6. Produce output/predictions_batch.csv con predizioni su tutto il dataset

Esecuzione:
    python src/pipeline_ml.py
"""

import os, re, pickle, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

# ── Preprocessing ──────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    """Minuscolo + rimozione punteggiatura e caratteri non alfabetici."""
    text = text.lower()
    text = re.sub(r"[^a-zàáâãäèéêëìíîïòóôõöùúûü\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Costruzione feature: title + body concatenati ──────────────────────────
def build_features(df: pd.DataFrame) -> pd.Series:
    return (df["title"] + " " + df["body"]).apply(preprocess)

# ── Costruzione pipeline sklearn ───────────────────────────────────────────
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            sublinear_tf=True,
        )),
        ("clf", MultinomialNB(alpha=0.5)),
    ])

# ── Valutazione e grafici ──────────────────────────────────────────────────
def evaluate(name: str, y_true, y_pred, labels: list, out_dir: str):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n{'='*50}")
    print(f"  Modello: {name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-macro : {f1:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predetto", fontsize=11)
    ax.set_ylabel("Reale",    fontsize=11)
    ax.set_title(f"Confusion Matrix – {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    fig_path = os.path.join(out_dir, f"confusion_{safe_name}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  [grafico] {fig_path}")
    return {"accuracy": round(acc, 4), "f1_macro": round(f1, 4)}

def bar_chart_f1(metrics: dict, out_dir: str):
    """Bar chart F1 per i due modelli."""
    nomi   = list(metrics.keys())
    valori = [metrics[n]["f1_macro"] for n in nomi]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(nomi, valori, color=["#4C72B0", "#DD8452"], width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-Score (macro)", fontsize=11)
    ax.set_title("F1-Score per modello", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, valori):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "f1_bar_chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [grafico] {path}")

# ── Top parole influenti (coefficienti TF-IDF × NB log-prob) ──────────────
def top_words(pipeline: Pipeline, label: str, classes: list, n: int = 5) -> list[str]:
    vectorizer = pipeline.named_steps["tfidf"]
    clf        = pipeline.named_steps["clf"]
    if label not in classes:
        return []
    idx = list(classes).index(label)
    log_probs = clf.feature_log_prob_[idx]
    top_idx   = np.argsort(log_probs)[-n:][::-1]
    feature_names = vectorizer.get_feature_names_out()
    return [feature_names[i] for i in top_idx]

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    os.makedirs("models",  exist_ok=True)
    os.makedirs("output",  exist_ok=True)
    os.makedirs("assets",  exist_ok=True)

    # 1. Carica dataset
    df = pd.read_csv("data/tickets_sintetici.csv")
    print(f"[OK] Dataset caricato: {len(df)} ticket")

    X = build_features(df)
    y_cat = df["category"]
    y_pri = df["priority"]

    # 2. Split 80/20
    (X_tr, X_te,
     yc_tr, yc_te,
     yp_tr, yp_te) = train_test_split(
        X, y_cat, y_pri,
        test_size=0.20, random_state=42, stratify=y_cat
    )
    print(f"     Train: {len(X_tr)}  |  Test: {len(X_te)}")

    # 3. Addestra modello categoria
    pipe_cat = build_pipeline()
    pipe_cat.fit(X_tr, yc_tr)
    yc_pred = pipe_cat.predict(X_te)

    # 4. Addestra modello priorità
    pipe_pri = build_pipeline()
    pipe_pri.fit(X_tr, yp_tr)
    yp_pred = pipe_pri.predict(X_te)

    # 5. Valutazione
    cat_labels = sorted(y_cat.unique())
    pri_labels = ["alta", "media", "bassa"]

    metrics = {}
    metrics["Categoria"] = evaluate("Categoria", yc_te, yc_pred, cat_labels, "assets")
    metrics["Priorità"]  = evaluate("Priorità",  yp_te, yp_pred, pri_labels, "assets")
    bar_chart_f1(metrics, "assets")

    # Salva metriche JSON (usato dalla dashboard)
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 6. Salva modelli
    with open("models/model_categoria.pkl", "wb") as f:
        pickle.dump(pipe_cat, f)
    with open("models/model_priorita.pkl", "wb") as f:
        pickle.dump(pipe_pri, f)
    print("\n[OK] Modelli salvati in models/")

    # 7. Top parole per ogni etichetta (salvate per la dashboard)
    top_words_dict = {}
    for cat in cat_labels:
        top_words_dict[cat] = top_words(pipe_cat, cat, pipe_cat.classes_)
    with open("models/top_words.json", "w") as f:
        json.dump(top_words_dict, f, ensure_ascii=False, indent=2)

    # 8. Predizioni batch sull'intero dataset
    X_all = build_features(df)
    df["pred_category"] = pipe_cat.predict(X_all)
    df["pred_priority"]  = pipe_pri.predict(X_all)
    batch_path = "output/predictions_batch.csv"
    df.to_csv(batch_path, index=False)
    print(f"[OK] Predizioni batch esportate: {batch_path}")

if __name__ == "__main__":
    main()
