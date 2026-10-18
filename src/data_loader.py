"""
data_loader.py
Tujuan : membaca dataset UCI "Default of Credit Card Clients" dan merapikannya.

Dua format file didukung:
  1) file asli UCI (.xls)  -> baris pertama berisi label X1..X23, baris kedua
     berisi nama kolom sebenarnya, jadi header ada di baris ke-2 (header=1).
  2) file .csv turunan (mis. UCI_Credit_Card.csv) -> header di baris pertama.
"""
from pathlib import Path

import pandas as pd

from config import TARGET


def load_raw(path):
    """Membaca file mentah tanpa mengubah isinya."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\n"
            "Unduh dataset dulu (lihat dataset/README.md) atau beri path "
            "lewat --data."
        )
    if path.suffix.lower() in (".xls", ".xlsx"):
        df = pd.read_excel(path, header=1)
    else:
        df = pd.read_csv(path)
        # jaga-jaga: csv hasil konversi xls kadang masih membawa baris X1..X23
        if "LIMIT_BAL" not in df.columns:
            df = pd.read_csv(path, header=1)
    return df


def standardize(df):
    """Menyeragamkan nama kolom, membuang ID, dan merapikan kode kategori."""
    df = df.copy()

    # 1) Nama kolom: hilangkan spasi/titik supaya konsisten antar versi file
    df.columns = (
        df.columns.astype(str).str.strip().str.replace(r"[ .]", "_", regex=True)
    )
    df = df.rename(columns={"default_payment_next_month": TARGET, "PAY_0": "PAY_1"})

    # 2) Kolom ID hanya penanda baris, bukan fitur -> dibuang
    if "ID" in df.columns:
        df = df.drop(columns="ID")

    # 3) Validasi: target harus ada dan hanya berisi 0/1
    if TARGET not in df.columns:
        raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan. Kolom: {list(df.columns)}")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError("Target harus berisi 0/1 saja.")

    # 4) Merapikan kode kategori.
    #    KEPUTUSAN PENULIS (cek ulang dengan codebook UCI):
    #    EDUCATION 0,5,6 tidak terdokumentasi -> digabung ke 4 ("others")
    #    MARRIAGE 0 tidak terdokumentasi      -> digabung ke 3 ("others")
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    return df.reset_index(drop=True)


def quality_report(df):
    """Ringkasan kualitas data (dicetak, tidak mengubah data)."""
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_missing_cells": int(df.isna().sum().sum()),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "class_counts": df[TARGET].value_counts().sort_index().to_dict(),
        "class_proportion": df[TARGET].value_counts(normalize=True).sort_index().round(4).to_dict(),
    }


def load_dataset(path):
    return standardize(load_raw(path))
