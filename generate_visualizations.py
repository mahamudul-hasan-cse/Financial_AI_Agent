"""Generate publication-quality visualizations for the NLP pipeline evaluation.

Creates 6 matplotlib charts in outputs/figures/:
  1. Confusion matrix heatmap (ML classifier)
  2. Learning curve (train/val accuracy vs training set size)
  3. Hyperparameter sensitivity (accuracy vs C, per ngram_range)
  4. Per-class F1 comparison (ML vs Keyword)
  5. NER per-entity-type (grouped bars: precision/recall/F1)
  6. Ablation study bar chart (degradation per condition)

Run directly: python generate_visualizations.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Ensure non-interactive backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "figures")


def _ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_confusion_matrix(save: bool = True) -> str:
    """1. Annotated confusion matrix heatmap for the ML classifier."""
    from evaluation import evaluate_intent_classifiers

    results = evaluate_intent_classifiers()
    ml = results.get("ml", {})
    if "confusion_matrix" not in ml:
        print("  [SKIP] ML classifier not available for confusion matrix.")
        return ""

    matrix = np.array(ml["confusion_matrix"])
    labels = ml["labels"]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Intent Classification — Confusion Matrix\n(TF-IDF + Logistic Regression)", fontsize=13)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    thresh = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black",
                    fontsize=10)

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    if save:
        _ensure_output_dir()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def plot_learning_curve(save: bool = True) -> str:
    """2. Learning curve — train/val accuracy vs training set fraction."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SkPipeline
    from intent_classifier import TRAINING_DATA, preprocess_text

    texts = [preprocess_text(t) for t, _ in TRAINING_DATA]
    labels = [l for _, l in TRAINING_DATA]

    fractions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    train_means, val_means = [], []
    train_stds, val_stds = [], []

    for frac in fractions:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        t_scores, v_scores = [], []

        for train_idx, val_idx in skf.split(texts, labels):
            # Sub-sample train set
            n_train = max(1, int(len(train_idx) * frac))
            sub_train_idx = train_idx[:n_train]

            train_texts = [texts[i] for i in sub_train_idx]
            train_labels = [labels[i] for i in sub_train_idx]
            val_texts = [texts[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]

            pipe = SkPipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000,
                                          stop_words="english", sublinear_tf=True)),
                ("clf", LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs")),
            ])
            pipe.fit(train_texts, train_labels)

            t_acc = sum(1 for p, t in zip(pipe.predict(train_texts), train_labels) if p == t) / len(train_labels)
            v_acc = sum(1 for p, t in zip(pipe.predict(val_texts), val_labels) if p == t) / len(val_labels)
            t_scores.append(t_acc)
            v_scores.append(v_acc)

        train_means.append(np.mean(t_scores))
        val_means.append(np.mean(v_scores))
        train_stds.append(np.std(t_scores))
        val_stds.append(np.std(v_scores))

    train_sizes = [int(f * len(texts) * 0.8) for f in fractions]  # approx per fold

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(train_sizes,
                     np.array(train_means) - np.array(train_stds),
                     np.array(train_means) + np.array(train_stds),
                     alpha=0.15, color="blue")
    ax.fill_between(train_sizes,
                     np.array(val_means) - np.array(val_stds),
                     np.array(val_means) + np.array(val_stds),
                     alpha=0.15, color="orange")
    ax.plot(train_sizes, train_means, "o-", color="blue", label="Training accuracy")
    ax.plot(train_sizes, val_means, "o-", color="orange", label="Validation accuracy")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve — TF-IDF + Logistic Regression")
    ax.legend(loc="lower right")
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "learning_curve.png")
    if save:
        _ensure_output_dir()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def plot_hyperparameter_sensitivity(save: bool = True) -> str:
    """3. Hyperparameter sensitivity — accuracy vs C, one line per ngram_range."""
    from intent_classifier import tune_hyperparameters

    results = tune_hyperparameters()
    all_results = results["all_results"]

    # Group by ngram_range
    ngram_groups: dict[str, tuple[list, list, list]] = {}
    for r in all_results:
        key = str(r["ngram_range"])
        if key not in ngram_groups:
            ngram_groups[key] = ([], [], [])
        ngram_groups[key][0].append(r["C"])
        ngram_groups[key][1].append(r["mean_accuracy"])
        ngram_groups[key][2].append(r["std_accuracy"])

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for (key, (cs, accs, stds)), color in zip(ngram_groups.items(), colors):
        ax.errorbar(cs, accs, yerr=stds, marker="o", label=f"ngram={key}",
                    capsize=3, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Regularization Strength (C) — log scale")
    ax.set_ylabel("Mean 5-Fold CV Accuracy")
    ax.set_title("Hyperparameter Sensitivity — C × ngram_range")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark best
    best = results["best_params"]
    ax.axvline(x=best["C"], color="red", linestyle="--", alpha=0.5,
               label=f"Best C={best['C']}")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "hyperparameter_sensitivity.png")
    if save:
        _ensure_output_dir()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def plot_per_class_f1(save: bool = True) -> str:
    """4. Per-class F1 comparison — ML vs Keyword side by side."""
    from evaluation import evaluate_intent_classifiers

    results = evaluate_intent_classifiers()
    kw = results.get("keyword", {})
    ml = results.get("ml", {})

    if "per_class" not in kw or "per_class" not in ml:
        print("  [SKIP] Classifiers not available for per-class F1 plot.")
        return ""

    labels = sorted(kw["per_class"].keys())
    kw_f1 = [kw["per_class"][l]["f1"] for l in labels]
    ml_f1 = [ml["per_class"][l]["f1"] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width / 2, kw_f1, width, label="Keyword", color="#FF9800", alpha=0.85)
    bars2 = ax.bar(x + width / 2, ml_f1, width, label="ML (TF-IDF + LogReg)", color="#2196F3", alpha=0.85)

    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score — ML vs Keyword Classifier")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "per_class_f1.png")
    if save:
        _ensure_output_dir()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def plot_ner_per_type(save: bool = True) -> str:
    """5. NER per-entity-type — grouped bars: precision/recall/F1."""
    from evaluation import evaluate_ner_per_type

    per_type = evaluate_ner_per_type()
    if not per_type:
        print("  [SKIP] No NER per-type results.")
        return ""

    types = list(per_type.keys())
    precision = [per_type[t]["precision"] for t in types]
    recall = [per_type[t]["recall"] for t in types]
    f1 = [per_type[t]["f1"] for t in types]

    x = np.arange(len(types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precision, width, label="Precision", color="#2196F3", alpha=0.85)
    ax.bar(x, recall, width, label="Recall", color="#FF9800", alpha=0.85)
    ax.bar(x + width, f1, width, label="F1", color="#4CAF50", alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title("NER Performance by Entity Type")
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=20, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "ner_per_type.png")
    if save:
        _ensure_output_dir()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def plot_ablation_chart(save: bool = True) -> str:
    """6. Ablation study — grouped bars showing metric degradation."""
    from ablation_study import run_ablation_study

    results = run_ablation_study()
    if not results:
        print("  [SKIP] Ablation study returned no results.")
        return ""

    conditions = [r["condition"] for r in results]
    metrics = ["intent_accuracy", "entity_recall", "sentiment_coverage"]
    metric_labels = ["Intent Accuracy", "Entity Recall", "Sentiment Coverage"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [r[metric] for r in results]
        ax.bar(x + i * width - width, values, width, label=label, color=color, alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title("Ablation Study — Component Impact on Pipeline Quality")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=25, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "ablation_study.png")
    if save:
        _ensure_output_dir()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def generate_all() -> list[str]:
    """Generate all 6 visualizations and return file paths."""
    _ensure_output_dir()
    paths = []

    charts = [
        ("Confusion Matrix", plot_confusion_matrix),
        ("Learning Curve", plot_learning_curve),
        ("Hyperparameter Sensitivity", plot_hyperparameter_sensitivity),
        ("Per-Class F1", plot_per_class_f1),
        ("NER Per-Type", plot_ner_per_type),
        ("Ablation Study", plot_ablation_chart),
    ]

    for name, fn in charts:
        print(f"  Generating: {name}...", end=" ")
        try:
            path = fn(save=True)
            if path:
                paths.append(path)
                print(f"OK → {os.path.basename(path)}")
            else:
                print("SKIPPED")
        except Exception as e:
            print(f"ERROR: {e}")

    return paths


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  NLP Pipeline — Visualization Generation")
    print("  Financial AI Agent — CSE495B")
    print("=" * 60 + "\n")

    paths = generate_all()

    print(f"\n  Generated {len(paths)} charts in {OUTPUT_DIR}/")
    for p in paths:
        print(f"    → {os.path.basename(p)}")
    print()
