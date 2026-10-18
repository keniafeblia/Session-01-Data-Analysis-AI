"""
run_experiments.py
Tujuan : menjalankan seluruh alur dari data mentah sampai hasil evaluasi.

Alur (sesuai konsep Sesi 1):
  1. Muat & rapikan data          -> data_loader.py
  2. EDA                          -> eda.py
  3. Pisahkan data uji (dibekukan, disentuh sekali)
  4. Bandingkan model pada data pengembangan dengan stratified CV (out-of-fold)
  5. Pilih "champion" berdasarkan PR-AUC, lalu tentukan threshold dari OOF
  6. Latih ulang champion pada seluruh data pengembangan -> uji SEKALI di data uji
  7. Simpan tabel, metrik, dan grafik ke results/

Cara menjalankan (dari folder Session-01-Data-Analysis-AI):
  python src/run_experiments.py --data "dataset/default of credit card clients.xls"
  python src/run_experiments.py --data dataset/UCI_Credit_Card.csv --quick   # uji cepat
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (DEFAULT_DATA_FILE, FIG_DIR, N_FOLDS, RESULTS_DIR,
                    TARGET, TEST_SIZE, THRESHOLD_BETA, RANDOM_STATE)
from data_loader import load_dataset
from eda import run_eda
from evaluate import (best_threshold, feature_importance, metrics_at, oof_proba,
                      plot_confusion, plot_importance, plot_pr_curve)
from models import get_configs


def main(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1-2. Data & EDA
    df = load_dataset(args.data)
    run_eda(df)

    X, y = df.drop(columns=TARGET), df[TARGET]

    # 3. Pisahkan data uji (stratified agar proporsi kelas sama)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    n_folds = 3 if args.quick else N_FOLDS
    pos_weight = float((y_dev == 0).sum() / (y_dev == 1).sum())
    print(f"[SPLIT] dev={len(X_dev)}  test={len(X_test)}  folds={n_folds}  pos_weight={pos_weight:.2f}")

    # 4. Bandingkan semua konfigurasi dengan out-of-fold prediction
    rows, oof_store, pipelines = [], {}, {}
    for name, model, strategy, pipe in get_configs(list(X.columns), pos_weight, args.quick):
        print(f"[CV] {name} ...")
        proba = oof_proba(pipe, X_dev, y_dev, n_folds)
        thr = best_threshold(y_dev, proba, THRESHOLD_BETA) if model != "dummy" else 0.5
        m05, mt = metrics_at(y_dev, proba, 0.5), metrics_at(y_dev, proba, thr)
        rows.append({
            "config": name, "model": model, "strategy": strategy,
            "accuracy@0.5": m05["accuracy"], "recall@0.5": m05["recall"],
            "precision@0.5": m05["precision"], "f1@0.5": m05["f1"],
            "roc_auc": m05["roc_auc"], "pr_auc": m05["pr_auc"],
            "tuned_threshold": thr, "recall@tuned": mt["recall"],
            "precision@tuned": mt["precision"], "f1@tuned": mt["f1"],
            "accuracy@tuned": mt["accuracy"],
        })
        oof_store[name], pipelines[name] = proba, pipe

    table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    table.round(4).to_csv(RESULTS_DIR / "cv_comparison.csv", index=False)
    print("\n[HASIL CV] (diurutkan menurut PR-AUC)")
    print(table.round(3).to_string(index=False))

    # 5. Pilih champion (tanpa dummy) dan threshold
    cand = table[table["model"] != "dummy"]
    champ = cand.iloc[0]
    champ_name, thr = champ["config"], float(champ["tuned_threshold"])
    print(f"\n[CHAMPION] {champ_name}  threshold={thr:.3f}")

    # 6. Latih ulang pada seluruh data pengembangan, uji SEKALI di data uji
    final = pipelines[champ_name].fit(X_dev, y_dev)
    test_proba = final.predict_proba(X_test)[:, 1]
    test_metrics = metrics_at(y_test, test_proba, thr)
    out = {
        "champion": champ_name,
        "selected_by": "PR-AUC out-of-fold pada data pengembangan",
        "threshold_beta": THRESHOLD_BETA,
        "test_metrics": test_metrics,
        "note": "Data uji dipakai satu kali; jangan ubah model setelah melihat angka ini.",
    }
    (RESULTS_DIR / "test_metrics.json").write_text(json.dumps(out, indent=2))
    print("[TEST]", json.dumps(test_metrics, indent=2))

    # 7. Grafik & fitur penting
    plot_pr_curve(y_test, test_proba, thr, FIG_DIR / "03_pr_curve_test.png")
    plot_confusion(y_test, test_proba, thr, FIG_DIR / "04_confusion_matrix_test.png")
    imp = feature_importance(final)
    if imp is not None:
        imp.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
        plot_importance(imp, FIG_DIR / "05_feature_importance.png")
    print("\nSelesai. Hasil tersimpan di:", RESULTS_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_DATA_FILE), help="path file dataset (.xls atau .csv)")
    ap.add_argument("--quick", action="store_true", help="mode uji cepat (model lebih kecil, 3 fold)")
    main(ap.parse_args())
