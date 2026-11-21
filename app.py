"""
Aplikasi Web Deteksi Berita Palsu (Fake News Detection)
Proyek UAS Big Data & Machine Learning

Cara kerja: saat pertama kali dibuka, aplikasi melatih model dari news.csv
(hanya beberapa detik), lalu hasilnya disimpan di cache.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

st.set_page_config(page_title="Deteksi Berita Palsu", page_icon="📰", layout="wide")

DATA_PATH = Path(__file__).parent / "news.csv"
RANDOM_STATE = 42


# ----------------------------------------------------------------------
# Preprocessing (sama persis dengan notebook Colab)
# ----------------------------------------------------------------------
def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)   # hapus URL
    t = re.sub(r"[^a-z\s]", " ", t)           # hapus angka & tanda baca
    t = re.sub(r"\s+", " ", t).strip()
    return t


@st.cache_resource(show_spinner="Melatih model dari news.csv, mohon tunggu sebentar...")
def train_all():
    df = pd.read_csv(DATA_PATH)
    n_raw = len(df)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df[df["text"].str.strip() != ""].drop_duplicates(subset="text").reset_index(drop=True)

    df["content"] = df["title"] + " " + df["text"]
    df["content_clean"] = df["content"].apply(clean_text)
    df["y"] = df["label"].map({"FAKE": 0, "REAL": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df["content_clean"], df["y"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df["y"])

    tfidf = TfidfVectorizer(stop_words="english", max_df=0.7, min_df=5,
                            max_features=50000, ngram_range=(1, 2))
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)

    models = {
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Naive Bayes": MultinomialNB(),
    }
    rows, cms = [], {}
    for name, m in models.items():
        m.fit(Xtr, y_train)
        pred = m.predict(Xte)
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_test, pred),
                     "Precision": precision_score(y_test, pred),
                     "Recall": recall_score(y_test, pred),
                     "F1-Score": f1_score(y_test, pred)})
        cms[name] = confusion_matrix(y_test, pred)

    metrics = pd.DataFrame(rows).set_index("Model").round(4)

    # contoh artikel dari data uji (untuk tombol "coba contoh")
    test_df = df.loc[X_test.index, ["title", "text", "label"]]

    return {"df": df, "n_raw": n_raw, "tfidf": tfidf, "models": models,
            "metrics": metrics, "cms": cms, "test_df": test_df,
            "n_train": len(X_train), "n_test": len(X_test)}


def predict(art, model_name, raw_text):
    """Kembalikan (label, skor_REAL 0-1, vektor tfidf)."""
    vec = art["tfidf"].transform([clean_text(raw_text)])
    m = art["models"][model_name]
    if hasattr(m, "predict_proba"):
        p_real = float(m.predict_proba(vec)[0, 1])
    else:  # LinearSVC: ubah jarak ke margin menjadi skor 0-1 (perkiraan)
        p_real = float(1 / (1 + np.exp(-m.decision_function(vec)[0])))
    return ("REAL" if p_real >= 0.5 else "FAKE"), p_real, vec


def top_words(art, vec, n=8):
    """Kata yang paling mendorong ke FAKE / REAL (berdasarkan Logistic Regression)."""
    lr = art["models"]["Logistic Regression"]
    names = np.array(art["tfidf"].get_feature_names_out())
    contrib = vec.multiply(lr.coef_[0]).toarray()[0]
    idx = np.where(contrib != 0)[0]
    if len(idx) == 0:
        return [], []
    order = idx[np.argsort(contrib[idx])]
    fake = [(names[i], contrib[i]) for i in order[:n] if contrib[i] < 0]
    real = [(names[i], contrib[i]) for i in order[::-1][:n] if contrib[i] > 0]
    return fake, real


# ----------------------------------------------------------------------
# Tampilan
# ----------------------------------------------------------------------
st.title("📰 Deteksi Berita Palsu")
st.caption("Proyek UAS Big Data & Machine Learning · Klasifikasi teks dengan TF-IDF")

if not DATA_PATH.exists():
    st.error("File **news.csv** tidak ditemukan. Letakkan news.csv di folder yang sama dengan app.py.")
    st.stop()

art = train_all()

with st.sidebar:
    st.header("⚙️ Pengaturan")
    best = art["metrics"]["F1-Score"].idxmax()
    model_name = st.selectbox("Pilih model", list(art["models"].keys()),
                              index=list(art["models"].keys()).index(best),
                              help="Model terbaik (F1 tertinggi) dipilih secara default.")
    st.metric("Accuracy model ini", f"{art['metrics'].loc[model_name, 'Accuracy']:.1%}")
    st.divider()
    st.markdown("**Dataset:** news.csv  \n"
                f"**Data latih:** {art['n_train']:,}  \n"
                f"**Data uji:** {art['n_test']:,}")

tab1, tab2, tab3 = st.tabs(["🔍 Prediksi", "📊 Performa Model", "ℹ️ Tentang Dataset"])

# ------------------------------ TAB 1: PREDIKSI ------------------------------
with tab1:
    st.write("Tempelkan teks berita **berbahasa Inggris** (dataset berbahasa Inggris), "
             "lalu klik **Analisis**.")

    def load_sample(label):
        row = art["test_df"][art["test_df"]["label"] == label].sample().iloc[0]
        st.session_state["inp_title"] = row["title"]
        st.session_state["inp_text"] = row["text"]
        st.session_state["true_label"] = label

    c1, c2, _ = st.columns([1, 1, 3])
    c1.button("🎲 Contoh artikel FAKE", on_click=load_sample, args=("FAKE",))
    c2.button("🎲 Contoh artikel REAL", on_click=load_sample, args=("REAL",))

    title = st.text_input("Judul berita", key="inp_title")
    text = st.text_area("Isi berita", height=250, key="inp_text")

    if st.button("🔍 Analisis", type="primary"):
        content = f"{title} {text}".strip()
        if len(content.split()) < 30:
            st.warning("Teks terlalu pendek. Masukkan minimal ±30 kata agar hasil lebih bermakna.")
        else:
            label, p_real, vec = predict(art, model_name, content)
            conf = p_real if label == "REAL" else 1 - p_real

            if label == "REAL":
                st.success(f"### ✅ Prediksi: REAL (berita asli)")
            else:
                st.error(f"### 🚩 Prediksi: FAKE (berita palsu)")
            st.progress(conf, text=f"Skor keyakinan: {conf:.1%}")
            if model_name == "Linear SVM":
                st.caption("Catatan: Linear SVM tidak menghasilkan probabilitas asli; "
                           "skor ini hanya perkiraan dari jarak ke garis pemisah.")

            true_label = st.session_state.get("true_label")
            if true_label and text == st.session_state.get("inp_text"):
                st.info(f"Label asli artikel contoh ini di dataset: **{true_label}**")

            fake_w, real_w = top_words(art, vec)
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Kata yang mengarah ke FAKE**")
                if fake_w:
                    st.dataframe(pd.DataFrame(fake_w, columns=["Kata", "Bobot"]).round(3),
                                 hide_index=True)
                else:
                    st.caption("Tidak ada.")
            with col_b:
                st.markdown("**Kata yang mengarah ke REAL**")
                if real_w:
                    st.dataframe(pd.DataFrame(real_w, columns=["Kata", "Bobot"]).round(3),
                                 hide_index=True)
                else:
                    st.caption("Tidak ada.")
            st.caption("Pengaruh kata dihitung dari model Logistic Regression.")

    st.warning("⚠️ Ini proyek akademik. Model mempelajari pola kata dari dataset berita politik AS "
               "(sekitar pemilu 2016), jadi **bukan alat pemeriksa fakta** yang andal untuk topik lain.")

# ------------------------------ TAB 2: PERFORMA ------------------------------
with tab2:
    st.subheader("Perbandingan Model pada Data Uji")
    st.dataframe(art["metrics"].style.format("{:.2%}").highlight_max(axis=0, color="#c6efce"))
    st.bar_chart(art["metrics"])

    st.subheader(f"Confusion Matrix: {model_name}")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ConfusionMatrixDisplay(art["cms"][model_name],
                           display_labels=["FAKE", "REAL"]).plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)
    st.caption("Baris = label asli, kolom = prediksi model.")

# ------------------------------ TAB 3: DATASET ------------------------------
with tab3:
    df = art["df"]
    st.subheader("Ringkasan Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Artikel awal", f"{art['n_raw']:,}")
    c2.metric("Setelah dibersihkan", f"{len(df):,}")
    c3.metric("Kolom dipakai", "title + text")

    st.markdown("**Distribusi label** (setelah duplikat dan teks kosong dibuang)")
    st.bar_chart(df["label"].value_counts())

    st.markdown("""
**Pipeline singkat**
1. Buang kolom indeks, teks kosong, dan duplikat
2. Gabungkan judul + isi, lalu bersihkan teks (lowercase, hapus URL/angka/tanda baca)
3. Encoding label: FAKE = 0, REAL = 1
4. Split 80:20 (stratified)
5. TF-IDF (unigram + bigram, maks. 50.000 fitur)
6. Latih Linear SVM, Logistic Regression, dan Naive Bayes

**Keterbatasan:** dataset didominasi politik AS 2016, dan model bisa belajar gaya penulisan
atau sumber berita, bukan kebenaran isi.
""")
    with st.expander("Lihat contoh data"):
        st.dataframe(df[["title", "label"]].sample(10, random_state=1), hide_index=True)
