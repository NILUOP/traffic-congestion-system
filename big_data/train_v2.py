"""
train.py — ML training pipeline for CARLA traffic congestion prediction.

Trains Logistic Regression, Random Forest, and XGBoost classifiers on the
processed splits, evaluates each model, and saves results + plots.

Leakage fix: traffic_density, density_bin, is_high_density, speed_x_density
are excluded from training because the congestion label is a deterministic
function of traffic_density + speed. Including them lets the model trivially
reverse-engineer the label rule instead of learning real traffic patterns.

USAGE:
    python train.py

REQUIRES:
    pip install scikit-learn xgboost pandas pyarrow matplotlib seaborn
"""

import os
import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay
)
from sklearn.multiclass      import OneVsRestClassifier

import xgboost as xgb

# =============================================================================
# CONFIG
# =============================================================================

PROCESSED_DIR   = "data/processed"
OUTPUT_DIR      = "data/models_clean"     # separate dir — preserves old results
PLOTS_DIR       = "data/models_clean/plots"

LABEL_NAMES     = ["Low", "Medium", "High"]   # index 0, 1, 2
RANDOM_SEED     = 42

# Features to DROP — they are direct inputs to the label definition.
# label_congestion() uses traffic_density + speed to assign the label,
# so any feature derived from traffic_density leaks the answer to the model.
LEAKAGE_FEATURES = [
    "traffic_density",   # directly used in label_congestion()
    "density_bin",       # derived from traffic_density
    "is_high_density",   # derived from traffic_density
    "speed_x_density",   # product of speed * traffic_density
]

# =============================================================================


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# =============================================================================
# LOAD
# =============================================================================

def load_split(name: str, feature_cols: list) -> tuple:
    path = os.path.join(PROCESSED_DIR, name)
    df   = pd.read_parquet(path)
    X    = df[feature_cols].astype(float).values
    y    = df["label"].astype(int).values
    return X, y


def load_data() -> tuple:
    section("STEP 1 — LOADING PROCESSED DATA")

    feature_path = os.path.join(PROCESSED_DIR, "feature_cols.txt")
    with open(feature_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    # Remove leakage features — these are derived from traffic_density which
    # is a direct input to the label function, making prediction trivially easy.
    feature_cols = [f for f in feature_cols if f not in LEAKAGE_FEATURES]
    print(f"  Leakage features removed : {LEAKAGE_FEATURES}")
    print(f"  Features used ({len(feature_cols)}): {feature_cols}")

    X_train, y_train = load_split("train", feature_cols)
    X_val,   y_val   = load_split("val",   feature_cols)
    X_test,  y_test  = load_split("test",  feature_cols)

    print(f"\n  Train : {X_train.shape[0]:>8,} rows x {X_train.shape[1]} features")
    print(f"  Val   : {X_val.shape[0]:>8,} rows")
    print(f"  Test  : {X_test.shape[0]:>8,} rows")

    # Class distribution check
    for split_name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        counts = np.bincount(y)
        dist   = "  ".join([f"{LABEL_NAMES[i]}={counts[i]:,}" for i in range(len(counts))])
        print(f"  {split_name:<6}: {dist}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


# =============================================================================
# EVALUATE
# =============================================================================

def evaluate(model, X, y, split_name: str) -> dict:
    y_pred      = model.predict(X)
    y_proba     = model.predict_proba(X)

    acc         = accuracy_score(y, y_pred)
    precision   = precision_score(y, y_pred, average="weighted", zero_division=0)
    recall      = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1          = f1_score(y, y_pred, average="weighted", zero_division=0)
    roc_auc     = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")

    return {
        "split":     split_name,
        "accuracy":  acc,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "roc_auc":   roc_auc,
        "y_pred":    y_pred,
        "y_proba":   y_proba,
    }


def print_metrics(results: dict, model_name: str):
    print(f"\n  [{model_name}] — {results['split']}")
    print(f"    Accuracy  : {results['accuracy']:.4f}")
    print(f"    Precision : {results['precision']:.4f}  (weighted)")
    print(f"    Recall    : {results['recall']:.4f}  (weighted)")
    print(f"    F1        : {results['f1']:.4f}  (weighted)")
    print(f"    ROC-AUC   : {results['roc_auc']:.4f}  (OvR weighted)")


# =============================================================================
# MODELS
# =============================================================================

def train_logistic_regression(X_train, y_train):
    section("STEP 2a — LOGISTIC REGRESSION")
    print("  Fitting (with StandardScaler)...")
    t0 = time.time()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter      = 1000,
            C             = 1.0,
            solver        = "lbfgs",
            multi_class   = "multinomial",
            random_state  = RANDOM_SEED,
            n_jobs        = -1,
        ))
    ])
    model.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    return model


def train_random_forest(X_train, y_train):
    section("STEP 2b — RANDOM FOREST")
    print("  Fitting...")
    t0 = time.time()

    model = RandomForestClassifier(
        n_estimators      = 200,
        max_depth         = 20,
        min_samples_leaf  = 10,
        max_features      = "sqrt",
        class_weight      = "balanced",
        random_state      = RANDOM_SEED,
        n_jobs            = -1,
    )
    model.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    section("STEP 2c — XGBOOST")
    print("  Fitting with early stopping on val set...")
    t0 = time.time()

    model = xgb.XGBClassifier(
        n_estimators        = 500,
        max_depth           = 6,
        learning_rate       = 0.05,
        subsample           = 0.8,
        colsample_bytree    = 0.8,
        min_child_weight    = 5,
        objective           = "multi:softprob",
        num_class           = 3,
        eval_metric         = "mlogloss",
        early_stopping_rounds = 20,
        random_state        = RANDOM_SEED,
        n_jobs              = -1,
        verbosity           = 0,
        device              = "cpu",   # change to "cuda" if you want GPU
    )
    model.fit(
        X_train, y_train,
        eval_set            = [(X_val, y_val)],
        verbose             = False,
    )
    best = model.best_iteration
    print(f"  Done in {time.time()-t0:.1f}s  (best iteration: {best})")
    return model


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(rf_model, xgb_model, feature_cols: list, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Feature importance", fontsize=14, fontweight="bold")

    for ax, model, title in [
        (axes[0], rf_model,  "Random Forest (Gini impurity)"),
        (axes[1], xgb_model, "XGBoost (gain)"),
    ]:
        importances = model.feature_importances_
        idx         = np.argsort(importances)
        colors      = ["#378ADD" if i != idx[-1] else "#D85A30" for i in idx]
        ax.barh(
            [feature_cols[i] for i in idx],
            importances[idx],
            color=colors, edgecolor="none", alpha=0.9
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Importance")
        ax.axvline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# =============================================================================
# CONFUSION MATRICES
# =============================================================================

def plot_confusion_matrices(results_by_model: dict, out_dir: str, y_test):
    n     = len(results_by_model)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Confusion matrices — test set", fontsize=14, fontweight="bold")

    for ax, (model_name, results) in zip(axes, results_by_model.items()):
        cm   = confusion_matrix(y_test, results["y_pred"])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(
            cm_pct, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            linewidths=0.5, ax=ax, cbar=True
        )
        ax.set_title(model_name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# =============================================================================
# ROC CURVES
# =============================================================================

def plot_roc_curves(models_and_results: list, X_test, y_test, out_dir: str):
    """
    models_and_results: list of (model_name, model, test_results)
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    y_bin = label_binarize(y_test, classes=[0, 1, 2])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("ROC curves (one-vs-rest) — test set", fontsize=14, fontweight="bold")

    colors = {
        "Logistic Regression": "#7F77DD",
        "Random Forest":       "#1D9E75",
        "XGBoost":             "#D85A30",
    }

    for class_idx, (ax, class_name) in enumerate(zip(axes, LABEL_NAMES)):
        for model_name, model, _ in models_and_results:
            y_proba = model.predict_proba(X_test)
            fpr, tpr, _ = roc_curve(y_bin[:, class_idx], y_proba[:, class_idx])
            auc_score   = auc(fpr, tpr)
            ax.plot(
                fpr, tpr,
                label=f"{model_name} (AUC={auc_score:.3f})",
                color=colors.get(model_name, "#888"),
                linewidth=1.8
            )

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax.set_title(f"Class: {class_name}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = os.path.join(out_dir, "roc_curves.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def print_comparison_table(all_results: dict):
    section("FINAL COMPARISON — TEST SET")

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    header  = f"  {'Model':<22}" + "".join([f"{m.upper():>12}" for m in metrics])
    print(header)
    print("  " + "-" * (22 + 12 * len(metrics)))

    best_vals = {m: max(r[m] for r in all_results.values()) for m in metrics}

    for model_name, results in all_results.items():
        row = f"  {model_name:<22}"
        for m in metrics:
            val  = results[m]
            mark = " *" if abs(val - best_vals[m]) < 1e-6 else "  "
            row += f"{val:>10.4f}{mark}"
        print(row)

    print(f"\n  * = best in column")


def save_comparison_csv(all_results: dict, out_dir: str):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []
    for model_name, results in all_results.items():
        row = {"model": model_name}
        row.update({m: round(results[m], 6) for m in metrics})
        rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "comparison.csv")
    df.to_csv(path, index=False)
    print(f"  Saved comparison → {path}")


def plot_comparison_bar(all_results: dict, out_dir: str):
    metrics     = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names = list(all_results.keys())
    x           = np.arange(len(metrics))
    width       = 0.25
    colors      = ["#7F77DD", "#1D9E75", "#D85A30"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        vals = [all_results[model_name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=model_name,
                      color=color, edgecolor="none", alpha=0.88)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — test set", fontsize=13, fontweight="bold")
    ax.legend()
    ax.axhline(0.8, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, label="0.8 target")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "model_comparison.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# =============================================================================
# PER-CLASS REPORT
# =============================================================================

def print_per_class_report(all_results: dict, y_test):
    section("PER-CLASS CLASSIFICATION REPORT — TEST SET")
    for model_name, results in all_results.items():
        print(f"\n  [{model_name}]")
        report = classification_report(
            y_test, results["y_pred"],
            target_names=LABEL_NAMES,
            digits=4
        )
        for line in report.split("\n"):
            print(f"    {line}")


# =============================================================================
# SAVE MODELS
# =============================================================================

def save_models(models: dict, out_dir: str):
    import pickle
    section("SAVING MODELS")
    for name, model in models.items():
        safe_name = name.lower().replace(" ", "_")
        path      = os.path.join(out_dir, f"{safe_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved {name} → {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)

    print("\nCARLA Traffic — ML Training Pipeline  [leakage-free]")
    print(f"  Excluded features : {LEAKAGE_FEATURES}")
    print(f"Processed dir : {PROCESSED_DIR}")
    print(f"Output dir    : {OUTPUT_DIR}")

    t_start = time.time()

    # --- Load ---
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data()

    # --- Train ---
    lr_model  = train_logistic_regression(X_train, y_train)
    rf_model  = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)

    models = {
        "Logistic Regression": lr_model,
        "Random Forest":       rf_model,
        "XGBoost":             xgb_model,
    }

    # --- Evaluate on val set (for reference) ---
    section("STEP 3 — VALIDATION SET METRICS")
    for model_name, model in models.items():
        val_results = evaluate(model, X_val, y_val, "val")
        print_metrics(val_results, model_name)

    # --- Evaluate on test set (final) ---
    section("STEP 4 — TEST SET METRICS")
    test_results = {}
    for model_name, model in models.items():
        results = evaluate(model, X_test, y_test, "test")
        test_results[model_name] = results
        print_metrics(results, model_name)

    # --- Reports ---
    print_per_class_report(test_results, y_test)
    print_comparison_table(test_results)

    # --- Plots ---
    section("STEP 5 — GENERATING PLOTS")
    plot_feature_importance(rf_model, xgb_model, feature_cols, PLOTS_DIR)
    plot_confusion_matrices(test_results, PLOTS_DIR, y_test)
    plot_roc_curves(
        [(n, m, test_results[n]) for n, m in models.items()],
        X_test, y_test, PLOTS_DIR
    )
    plot_comparison_bar(test_results, PLOTS_DIR)

    # --- Save ---
    save_models(models, OUTPUT_DIR)
    save_comparison_csv(test_results, OUTPUT_DIR)

    section("DONE")
    print(f"  Total time : {time.time()-t_start:.1f}s")
    print(f"  Models     → {OUTPUT_DIR}/")
    print(f"  Plots      → {PLOTS_DIR}/\n")


if __name__ == "__main__":
    main()