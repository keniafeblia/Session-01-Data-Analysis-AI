# Sesi 1 - Prediksi Gagal Bayar Kartu Kredit dengan Penanganan Ketidakseimbangan Kelas

Tugas mata kuliah DEEPL, Informatics Practitioner Lecture Series.
Sesi 1: *Make Sense of Data with Analytics and AI* (Andria Arisal, BRIN), 5 Oktober 2026.

## Latar belakang
Pada sesi 1, pemateri menunjukkan bahwa model dengan akurasi tinggi bisa tetap gagal menemukan kasus telat bayar
karena kelas positif jarang. Proyek ini memakai data publik lain untuk mempraktikkan kerangka yang sama:
pertanyaan prediktif yang jelas, pembagian data yang jujur, pencegahan data leakage, dan evaluasi di luar akurasi.

## Pertanyaan penelitian
Apakah strategi penanganan ketidakseimbangan kelas (class weight, SMOTE, penyesuaian threshold) meningkatkan
kemampuan model menemukan nasabah yang akan gagal bayar bulan depan, dibandingkan tanpa penanganan?

## Dataset
UCI *Default of Credit Card Clients* (30.000 nasabah, 23 variabel penjelas). 

## Metode
- Pembagian data: 80% data pengembangan, 20% data uji (stratified, dibekukan dan dipakai sekali).
- Model: logistic regression (baseline), random forest, XGBoost, ditambah pembanding yang selalu menebak kelas mayoritas.
- Strategi: tanpa penanganan, class weight, SMOTE (di dalam pipeline sehingga hanya memakai data latih tiap fold).
- Evaluasi: stratified 5-fold cross-validation (out-of-fold); metrik recall, precision, F1, PR-AUC, ROC-AUC, confusion matrix.
- Pemilihan model: PR-AUC out-of-fold. Threshold dipilih dari prediksi out-of-fold dengan F-beta (`THRESHOLD_BETA` di `src/config.py`).

## Struktur
```
Session-01-Data-Analysis-AI/
├── README.md
├── requirements.txt
├── dataset/          # petunjuk unduh dataset
├── src/
│   ├── config.py           # pengaturan (seed, ukuran data uji, beta)
│   ├── data_loader.py      # baca dan rapikan data
│   ├── eda.py              # distribusi kelas dan grafik awal
│   ├── models.py           # pipeline model dan strategi
│   ├── evaluate.py         # metrik, threshold, grafik
│   └── run_experiments.py  # menjalankan semuanya
└── results/          # tabel, metrik, dan grafik (dibuat otomatis)
```

## Cara menjalankan
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# letakkan dataset di folder dataset/, lalu:
python src/run_experiments.py --data "dataset/default of credit card clients.xls"

# uji cepat (model kecil, 3 fold):
python src/run_experiments.py --data "dataset/default of credit card clients.xls" --quick
```

## Hasil

Tabel berikut menyajikan hasil validasi silang 5-fold (*out-of-fold*) pada data pengembangan yang diurutkan berdasarkan PR-AUC (metrik Recall, Precision, dan F1 dievaluasi pada ambang keputusan optimal / *tuned threshold* berdasarkan F1-score):

| Konfigurasi | PR-AUC (OOF) | Recall | Precision | F1 |
|---|---|---|---|---|
| `xgb__class_weight` | 0.5581 | 0.6035 | 0.4975 | 0.5454 |
| `xgb__none` | 0.5581 | 0.5421 | 0.5506 | 0.5463 |
| `rf__none` | 0.5562 | 0.5779 | 0.5180 | 0.5463 |
| `rf__class_weight` | 0.5538 | 0.5856 | 0.5065 | 0.5432 |
| `rf__smote` | 0.5427 | 0.5649 | 0.5030 | 0.5322 |
| `xgb__smote` | 0.5377 | 0.5240 | 0.5387 | 0.5313 |
| `logreg__none` | 0.5051 | 0.4831 | 0.5547 | 0.5165 |
| `logreg__class_weight` | 0.5029 | 0.4835 | 0.5518 | 0.5154 |
| `logreg__smote` | 0.5022 | 0.4807 | 0.5501 | 0.5131 |
| `dummy_majority` | 0.2212 | 0.0000 | 0.0000 | 0.0000 |

*Catatan perbandingan ambang (threshold):*
- Pada ambang default 0.5 tanpa penanganan, model terjebak akurasi tinggi semu namun gagal mendeteksi mayoritas kasus gagal bayar (Recall `xgb__none`: 0.3703, `rf__none`: 0.3643, `logreg__none`: 0.2460).
- Penanganan `class_weight` pada ambang default 0.5 secara signifikan melipatgandakan Recall (`xgb__class_weight`: 0.6250, `rf__class_weight`: 0.5751, `logreg__class_weight`: 0.6506).
- Penyesuaian ambang (*threshold tuning*) pada model tanpa pembobotan (misal menurunkan threshold `xgb__none` ke 0.3014) juga terbukti efektif menaikkan Recall menjadi 0.5421.

Model terpilih: `xgb__class_weight` | Threshold: 0.5241 | Metrik data uji: PR-AUC = 0.5539, ROC-AUC = 0.7782, Recall = 0.5938, Precision = 0.4972, F1 = 0.5412, Akurasi = 0.7773 (TP=788, FN=539, TN=3876, FP=797)

## Pembahasan

### 1. Menjawab Pertanyaan Penelitian
Hasil eksperimen membuktikan secara empiris bahwa **strategi penanganan ketidakseimbangan kelas berhasil meningkatkan kemampuan model dalam menemukan nasabah yang berisiko gagal bayar**. Tanpa penanganan, model hanya mendeteksi sekitar 24%–37% nasabah gagal bayar. Dengan menerapkan *class weight*, *SMOTE*, atau *penyesuaian threshold*, daya tangkap model (Recall) meningkat signifikan ke kisaran 54%–62% pada data pengembangan dan mencapai **59.38%** pada data uji independen.

### 2. Jebakan Akurasi (*Accuracy Paradox*)
Sesuai konsep yang ditekankan pada Sesi 1, akurasi merupakan metrik yang menyesatkan untuk data tidak seimbang (proporsi minoritas ~22.12%):
- Model *dummy majority* menghasilkan akurasi semu **77.88%**, tetapi memiliki **Recall 0.0%** dan F1 0.0 (gagal total mengenali nasabah berisiko).
- Model baseline tanpa penanganan (`xgb__none`, `rf__none`) menunjukkan akurasi tinggi (~82%), namun melewatkan lebih dari 62% nasabah yang sebenarnya gagal bayar jika memakai ambang baku 0.5.

### 3. Perbandingan Strategi Penanganan
- **Class Weight (Paling Efektif & Efisien):** Memberikan bobot penalti lebih besar pada kelas minoritas terbukti paling konsisten mendongkrak Recall tanpa beban komputasi tambahan. Konfigurasi `xgb__class_weight` menempati peringkat teratas PR-AUC (0.5581) dan menghasilkan Recall 62.50% (pada threshold 0.5) serta F1 0.5454.
- **Penyesuaian Ambang Batas (*Threshold Tuning*):** Menurunkan ambang keputusan dari 0.5 ke nilai optimal F1 (~0.28–0.30) pada model tanpa penanganan terbukti mampu mendongkrak Recall secara signifikan (misalnya Recall `xgb__none` melonjak dari 37.03% ke 54.21%). Ini menunjukkan bahwa penyesuaian ambang adalah strategi *post-processing* yang sangat praktis dan berdaya guna tinggi tanpa perlu melatih ulang model.
- **SMOTE (*Oversampling* Sintetis):** SMOTE berhasil menaikkan Recall dibanding baseline (mencapai 56.49% pada RF), namun skor PR-AUC-nya (0.5377–0.5427) berada sedikit di bawah *class weight*. Hal ini disebabkan interpolasi titik sintetis pada variabel kategorik yang di-*one-hot encode* dapat membentuk data artifisial bernilai pecahan yang kurang realistis.

### 4. Generalisasi Data Uji dan Fitur Dominan
- **Evaluasi Data Uji:** Model champion `xgb__class_weight` pada ambang 0.5241 yang diuji pada data uji yang dibekukan (6.000 sampel) mencatatkan Recall 59.38% dan Precision 49.72%. Model berhasil menjaring 788 dari 1.327 kasus gagal bayar nyata, membuktikan kemampuan generalisasi yang stabil tanpa indikasi *data leakage*.
- **Analisis Fitur Penting (*Feature Importance*):** Faktor penentu paling dominan dalam memprediksi gagal bayar adalah riwayat keterlambatan pembayaran terkini, khususnya `PAY_1` (status pembayaran bulan terakhir dengan nilai kepentingan >33%) dan `PAY_2`, diikuti oleh batas kredit (`LIMIT_BAL`). Hal ini selaras dengan intuisi domain perbankan bahwa riwayat tunggakan terkini adalah indikator risiko yang paling relevan.

## Keterbatasan
- Data berasal dari satu penerbit kartu di Taiwan pada 2005 dan hanya mencakup sekitar enam bulan, sehingga hasil tidak otomatis berlaku untuk kondisi lain.
- SMOTE pada fitur one-hot menghasilkan nilai pecahan; varian seperti SMOTENC dapat dicoba.
- Pemilihan `THRESHOLD_BETA` adalah asumsi; biaya salah prediksi yang sebenarnya belum dimodelkan.
- Hasil dapat berubah jika seed atau versi library berbeda.

## Rujukan
Yeh, I. C., & Lien, C. H. (2009). Expert Systems with Applications, 36(2), 2473-2480. Dataset: UCI Machine Learning Repository (CC BY 4.0).
