"""
evaluate.py
Tujuan : menghitung metrik, memilih threshold, dan membuat grafik evaluasi.
Prinsip: akurasi saja menyesatkan pada data tidak seimbang, jadi yang dilaporkan
         adalah recall, precision, F1, PR-AUC, ROC-AUC, dan confusion matrix.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from config import FIG_DIR, RANDOM_STATE


def oof_proba(pipeline, X, y, n_folds):
    """Probabilitas out-of-fold: tiap baris diprediksi oleh model yang TIDAK pernah melihatnya."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]


def metrics_at(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "threshold": float(thr),
        "accuracy": float((tp + tn) / len(y)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_threshold(y, proba, beta):
    """Threshold yang memaksimalkan F-beta (beta>1 = lebih mementingkan recall)."""
    prec, rec, thr = precision_recall_curve(y, proba)
    prec, rec = prec[:-1], rec[:-1]  # selaraskan panjang dengan thresholds
    b2 = beta ** 2
    denom = b2 * prec + rec
    fbeta = np.divide((1 + b2) * prec * rec, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thr[int(np.argmax(fbeta))])


def plot_pr_curve(y, proba, thr, path):
    prec, rec, _ = precision_recall_curve(y, proba)
    m = metrics_at(y, proba, thr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, label=f"PR-AUC = {m['pr_auc']:.3f}")
    ax.scatter([m["recall"]], [m["precision"]], color="red", zorder=3, label=f"threshold = {thr:.2f}")
    ax.axhline(float(np.mean(y)), linestyle="--", color="gray", label="baseline (proporsi kelas positif)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (data uji)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y, proba, thr, path):
    pred = (proba >= thr).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=["Tidak gagal", "Gagal bayar"], ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix (data uji, threshold {thr:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def feature_importance(fitted_pipeline, top_n=15):
    """Pentingnya fitur dari model akhir (feature_importances_ atau |koefisien|)."""
    names = fitted_pipeline.named_steps["prep"].get_feature_names_out()
    est = fitted_pipeline.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    elif hasattr(est, "coef_"):
        values = np.abs(est.coef_).ravel()
    else:
        return None
    df = pd.DataFrame({"feature": names, "importance": values})
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def plot_importance(df, path):
    d = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(d["feature"], d["importance"])
    ax.set_title("Fitur terpenting (model terpilih)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
