# Tubes2AI-SatuWardaEmpatBeban
Deskripsi singkat
-----------------

Berikut pipeline lengkap tugas besar 2 AI, meliputi:
- **Data cleaning & preprocessing**: penanganan missing value, encoding, normalisasi, dan transformasi data.
- **Feature engineering**: seleksi dan pembuatan fitur baru untuk meningkatkan performa model.
- **Modeling**: implementasi dan evaluasi 3 model utama, baik dari awal (scratch) maupun pustaka scikit-learn:
	- Decision Tree Learning (C4.5)
	- Logistic Regression
	- Support Vector Machine (SVM)
- **Evaluasi**: membandingkan performa ketiga model menggunakan macro F1-score dan confusion matrix, dengan visualisasi dan analisis di notebook.

Semua tahapan terdokumentasi di notebook (`src/full-notebook.ipynb`) dan skrip Python pada folder `src/models/` dan `src/test/`.

Isi utama
---------
- `src/full-notebook.ipynb` : Notebook utama berisi seluruh pipeline (preprocessing, feature engineering, training, evaluasi, visualisasi).
- `src/models/decision_tree_learning.py` : Implementasi C4.5 dari awal (fit, predict, save/load, visualize).
- `src/models/LogRegression.py` : Implementasi Logistic Regression dari awal.
- `src/models/SVM.py` : Implementasi SVM dari awal.
- `src/models/LogReg_visualization_test.py`, `src/models/SVM_visualization.py` : Visualisasi hasil model.
- `src/data/` : Folder berisi `train.csv` dan `test.csv` contoh.
- `src/test/` : Skrip pengujian dan demo (mis. `test_c45_dummy.py`).
- `main.py` : Entrypoint sederhana (opsional) untuk eksperimen cepat.

Requirements / Prasyarat
------------------------
- Python 3.8+ (direkomendasikan 3.10/3.11)
- Virtual environment (venv) atau conda
- Paket Python: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `jupyter` (opsional)

Langkah setup (PowerShell, Windows)
-----------------------------------
1. Buat dan aktifkan virtual environment:

```powershell
python -m venv .venv
; .\.venv\Scripts\Activate.ps1
```

2. Pasang dependensi (contoh):

```powershell
pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn jupyter
```

atau jika Anda memiliki `requirements.txt`:

```powershell
pip install -r requirements.txt
```

Cara menjalankan
---------------
- Jalankan skrip pengujian/demo (contoh):

```powershell
cd src\test
python test_c45_dummy.py
```

- Untuk melatih menggunakan skrip utama: (sesuaikan argumen jika ada)

```powershell
cd src\models
python train_with_dataset.py
```

- Jupyter notebook: buka notebook yang tersedia di `src/full-notebook.ipynb` atau `src/notebook.ipynb`:

```powershell
jupyter notebook src\full-notebook.ipynb
```

Hasil dan artefak
-----------------
- Model yang disimpan (pickle) dan visualisasi pohon biasanya disimpan di `src/models/` setelah menjalankan skrip training atau test. Contoh: `src/models/tree_top3.png`, `src/models/tree_dummy.png`.

Evaluasi
--------
Evaluasi model menggunakan metrik yang sesuai untuk dataset (mis. macro F1-score untuk kasus ketidakseimbangan kelas). Lihat notebook untuk tabel perbandingan precision/recall/F1 dan confusion matrix.

Pembagian tugas (Kelompok 24)
---------------------------

| No | Nama | NIM | Tugas / Peran |
|----|------|-----|---------------|
| 1  | Wardatul Khoiroh | 13523001 | Implementasi C4.5 (scratch) + bonusnya, laporan |
| 2  | Raka Dafa | 13523018 | Implementasi SVM + bonusnya, laporan |
| 3  | M Fithra | 13523049 | Data Cleaning dan Preprocessing (notebook), laporan |
| 4  | Ahsan Malik Al Farisi | 13523074 | Implementasi Logreg + bonusnya, laporan |
| 5  | Farrel Athalla Putra | 13523118 | Data Cleaning dan Preprocessing (notebook), laporan |
