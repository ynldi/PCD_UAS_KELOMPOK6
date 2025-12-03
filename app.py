import os
import json
import zipfile

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# =========================
#  CONFIG HALAMAN
# =========================
st.set_page_config(
    page_title="Fruits & Vegetables Classifier",
    page_icon="🥕",
    layout="centered"
)


# =========================
#  EKSTRAK MODEL DARI ZIP
# =========================
def extract_model_zip():
    zip_path = "cnn_model.zip"   # file ZIP yg kamu upload di GitHub
    extract_to = "./"

    # Jika file model sudah ada, tidak perlu extract lagi
    if os.path.exists("cnn_model.h5"):
        return

    # Jika ZIP tidak ada → error
    if not os.path.exists(zip_path):
        st.error(f"File ZIP tidak ditemukan: {zip_path}")
        st.stop()

    # Ekstrak ZIP
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        st.success("📦 Model berhasil diekstrak dari cnn_model.zip")
    except Exception as e:
        st.error(f"Gagal mengekstrak ZIP: {e}")
        st.stop()


# =========================
#  LOAD MODEL & KELAS
# =========================
@st.cache_resource
def load_model_and_classes():
    # Ekstrak dulu ZIP agar cnn_model.h5 tersedia
    extract_model_zip()

    model_path = "cnn_model.h5"
    if not os.path.exists(model_path):
        st.error(f"cnn_model.h5 tidak ditemukan setelah extract ZIP")
        st.stop()

    # Load model Keras
    model = load_model(model_path)

    # Load classes.json
    classes_path = "classes.json"
    idx_to_class = None

    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            data = json.load(f)
        idx_to_class = {int(k): v for k, v in data.items()}
    else:
        idx_to_class = {
            0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage',
            5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn',
            10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger', 14: 'grapes',
            15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango',
            20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas',
            25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish',
            29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato',
            33: 'tomato', 34: 'turnip', 35: 'watermelon'
        }

    # Tentukan ukuran input
    input_shape = model.input_shape
    if len(input_shape) == 4:
        img_size = (input_shape[1], input_shape[2])
    else:
        img_size = (128, 128)

    return model, idx_to_class, img_size


model, idx_to_class, IMG_SIZE = load_model_and_classes()


# =========================
#  PREPROCESS DAN PREDIKSI
# =========================
def preprocess_image(image: Image.Image, img_size):
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(img_size)
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict(image: Image.Image):
    img = preprocess_image(image, IMG_SIZE)
    preds = model.predict(img)[0]
    pred_idx = int(np.argmax(preds))
    pred_label = idx_to_class.get(pred_idx, "Unknown")
    return pred_label, preds


# =========================
#  UI STREAMLIT
# =========================
if os.path.exists("banner.png"):
    st.image("banner.png", use_container_width=True)

st.title("🥕 Fruits and Vegetables Classification")
st.write("Upload gambar buah atau sayur untuk diprediksi.")


# Sidebar
st.sidebar.header("Pengaturan")
show_prob_table = st.sidebar.checkbox("Tampilkan tabel probabilitas", value=True)
st.sidebar.markdown("---")
st.sidebar.write(f"Model input size: **{IMG_SIZE[0]}×{IMG_SIZE[1]}**")


# Upload
uploaded_file = st.file_uploader(
    "Pilih gambar (JPG/PNG):",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    if st.button("🔍 Prediksi"):
        with st.spinner("Sedang memproses..."):
            label, probs = predict(image)
            confidence = float(np.max(probs) * 100.0)

        st.subheader("Hasil Prediksi")
        st.markdown(
            f"**Label:** `{label}`\n\n"
            f"**Confidence:** `{confidence:.2f}%`"
        )

        if show_prob_table:
            st.markdown("### Probabilitas per Kelas")

            sorted_indices = sorted(idx_to_class.keys())
            classes = [idx_to_class[i] for i in sorted_indices]
            prob_values = [float(probs[i] * 100.0) for i in sorted_indices]

            df = pd.DataFrame({
                "Class": classes,
                "Probability (%)": [round(p, 2) for p in prob_values]
            })

            st.dataframe(df)

            chart_df = df.set_index("Class")
            st.bar_chart(chart_df["Probability (%)"])
else:
    st.info("Silakan upload gambar terlebih dahulu 👆")
