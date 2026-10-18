"""
config.py
Tujuan : menyimpan semua pengaturan di satu tempat supaya mudah diubah
         dan hasil eksperimen bisa direproduksi (poin reproducibility di Sesi 1).
"""
from pathlib import Path

# ---- Lokasi folder ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

# Nama file default (hasil unduhan dari UCI, lihat dataset/README.md)
DEFAULT_DATA_FILE = DATA_DIR / "default of credit card clients.xls"

# ---- Reproducibility -------------------------------------------------------
RANDOM_STATE = 42

# ---- Pembagian data --------------------------------------------------------
TEST_SIZE = 0.20      # data uji dibekukan, hanya dipakai SEKALI di akhir
N_FOLDS = 5           # stratified k-fold pada data pengembangan (dev set)

# ---- Pemilihan ambang keputusan (threshold) --------------------------------
# beta = 1 -> F1 (precision dan recall sama penting).
# beta > 1 -> lebih menekankan recall (menemukan nasabah berisiko), tetapi
# jika terlalu besar threshold cenderung turun sampai hampir semua nasabah
# diprediksi berisiko (precision anjlok). Cek hasilnya di results/cv_comparison.csv.
# CATATAN: nilai ini adalah ASUMSI penulis, bukan dari dataset. Jika ingin memakai
# biaya salah prediksi yang nyata, ganti pemilihan threshold dengan biaya (pengembangan).
THRESHOLD_BETA = 1.0

# ---- Kolom -----------------------------------------------------------------
TARGET = "default"
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
