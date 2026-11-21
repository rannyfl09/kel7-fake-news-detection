# Deteksi Berita Palsu (Fake News Detection)

Proyek UAS Big Data & Machine Learning. Aplikasi web Streamlit yang memprediksi
apakah sebuah artikel berita berbahasa Inggris **REAL** atau **FAKE**
(TF-IDF + Linear SVM / Logistic Regression / Naive Bayes).

## Struktur folder
```
├── app.py            # aplikasi Streamlit
├── requirements.txt  # daftar library
├── news.csv          # dataset (WAJIB ada di sini)
└── README.md
```

## Menjalankan di komputer sendiri
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy gratis (Streamlit Community Cloud)
1. Upload semua file di atas ke repository GitHub (termasuk `news.csv`).
2. Buka https://share.streamlit.io lalu login dengan akun GitHub.
3. Klik **Create app** → pilih repository, branch `main`, dan file `app.py`.
4. Klik **Deploy**. Link web akan muncul setelah beberapa menit.

Catatan: model dilatih otomatis dari `news.csv` saat aplikasi pertama dibuka
(beberapa detik), lalu disimpan di cache.
