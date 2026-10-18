"""
eda.py
Tujuan : Exploratory Data Analysis singkat. Fokusnya satu pertanyaan:
         seberapa tidak seimbang kelas target? (inti kegagalan "akurasi tinggi"
         pada contoh sesi 1).
"""
import json

import matplotlib
matplotlib.use("Agg")  # supaya jalan di server/tanpa layar
import matplotlib.pyplot as plt

from config import FIG_DIR, RESULTS_DIR, TARGET
from data_loader import quality_report


def run_eda(df):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    report = quality_report(df)
    (RESULTS_DIR / "data_quality.json").write_text(json.dumps(report, indent=2))
    print("[EDA] Ringkasan kualitas data:")
    print(json.dumps(report, indent=2))

    # Grafik 1: distribusi kelas
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar(["Tidak gagal bayar (0)", "Gagal bayar (1)"], counts.values)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set_title("Distribusi kelas target")
    ax.set_ylabel("Jumlah nasabah")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_class_distribution.png", dpi=150)
    plt.close(fig)

    # Grafik 2: tingkat gagal bayar menurut status pembayaran terbaru (PAY_1)
    if "PAY_1" in df.columns:
        rate = df.groupby("PAY_1")[TARGET].mean()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(rate.index.astype(str), rate.values)
        ax.set_xlabel("PAY_1 (status pembayaran bulan terakhir)")
        ax.set_ylabel("Proporsi gagal bayar")
        ax.set_title("Proporsi gagal bayar menurut PAY_1")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "02_default_rate_by_pay1.png", dpi=150)
        plt.close(fig)

    return report
