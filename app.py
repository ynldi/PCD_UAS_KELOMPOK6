import os
import json
import zipfile

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

# ======================
#   PATCH KOMPATIBILITAS MODEL .H5
# ======================
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer

# Paksa Keras ke compatibility mode
try:
    tf.keras.utils.set_keras_version("2")
except:
    pass

# Matikan strict config Keras 3
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"

# Fallback custom layer supaya load_model tidak error
class DummyLayer(Layer):
    def call(self, inputs):
        return inputs

CUSTOM_OBJECTS = {
    "Functional": tf.keras.Model,
    "Model": tf.keras.Model,
    "Sequential": tf.keras.Sequential,
    "DummyLayer": DummyLayer
}

# =========================
#  CONFIG HALAMAN
# =========================
st.set_page_config(
    page_title="Fruits & Vegetables Classifier",
    page_icon="🥕",
    layout="centered"
)

# =========================
#  EKSTRAK MODEL ZIP
# =========================
def extract_model_zip():
    zip_path = "cnn_model.zip"
    extract_to = "./"

    if os.path.exists("cnn_model.h5"):
        return

    if not os.path.exists(zip_path):
        st.error(f"File ZIP tidak ditemukan: {zip_path}")
        st.stop()

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        st.success("📦 Model berhasil diekstrak.")
    except Exception as e:
        st.error(f"Gagal ekstrak ZIP: {e}")
        st.stop()

# =========================
#  LOAD MODEL & KELAS
# =========================
@st.cache_resource
def load_model_and_classes():
    extract_model_zip()
    model_path = "cnn_model.h5"

    if not os.path.exists(model_path):
        st.error("cnn_model.h5 tidak ditemukan setelah extract ZIP.")
        st.stop()

    # COMPAT LOAD MODEL
    try:
        model = load_model(model_path, compile=False)
    except Exception:
        model = load_model(model_path, compile=False, custom_objects=CUSTOM_OBJECTS)

    # Load classes.json (opsional)
    classes_path = "classes.json"
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

    input_shape = model.input_shape
    if len(input_shape) == 4:
        img_size = (input_shape[1], input_shape[2])
    else:
        img_size = (128, 128)

    return model, idx_to_class, img_size

model, idx_to_class, IMG_SIZE = load_model_and_classes()

# =========================
#  PREPROCESS & PREDIKSI
# =========================
def preprocess_image(image, img_size):
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Crop tengah agar objek lebih fokus
    w, h = image.size
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    right = left + min_side
    bottom = top + min_side
    image = image.crop((left, top, right, bottom))

    # Resize ke ukuran input model
    image = image.resize(img_size)

    # Normalisasi brightness agar lebih konsisten
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)

    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(img):
    x = preprocess_image(img, IMG_SIZE)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    label = idx_to_class.get(idx, "Unknown")
    return label, preds

# =========================
#  UI STREAMLIT
# =========================
if os.path.exists("banner.png"):
    st.image("banner.png", use_container_width=True)

st.title("🥕 Fruits and Vegetables Classification")
st.write("Pilih metode input: upload file atau ambil foto dengan kamera.")

# Sidebar menu terpisah
st.sidebar.header("📂 Menu Upload")
show_prob_upload = st.sidebar.checkbox("Tampilkan probabilitas Upload", True)

st.sidebar.header("📸 Menu Kamera")
show_prob_camera = st.sidebar.checkbox("Tampilkan probabilitas Kamera", True)

st.sidebar.markdown("---")
st.sidebar.write(f"Model input size: **{IMG_SIZE[0]}×{IMG_SIZE[1]}**")

# --- Menu Upload File ---
st.header("📂 Upload Gambar")
uploaded = st.file_uploader("Upload gambar buah/sayur:", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Gambar diupload", use_container_width=True)

    if st.button("🔍 Prediksi dari Upload"):
        with st.spinner("Memproses..."):
            label, probs = predict(img)
            conf = float(np.max(probs) * 100)

        st.subheader("Hasil Prediksi (Upload)")
        st.write(f"**Label:** `{label}`")
        st.write(f"**Confidence:** `{conf:.2f}%`")

        if show_prob_upload:
            df = pd.DataFrame({
                "Class": list(idx_to_class.values()),
                "Probability": [round(float(p) * 100, 2) for p in probs]
            })
            st.dataframe(df)
            st.bar_chart(df.set_index("Class"))

# --- Menu Kamera ---
st.header("📸 Ambil Foto dengan Kamera")
camera_file = st.camera_input("Klik 'Open Camera' lalu ambil gambar")

if camera_file:
    img = Image.open(camera_file)
    st.image(img, caption="Foto dari kamera", use_container_width=True)

    if st.button("🔍 Prediksi dari Kamera"):
        with st.spinner("Memproses..."):
            label, probs = predict(img)
            conf = float(np.max(probs) * 100)

        st.subheader("Hasil Prediksi (Kamera)")
        st.write(f"**Label:** `{label}`")
        st.write(f"**Confidence:** `{conf:.2f}%`")

        if show_prob_camera:
            df = pd.DataFrame({
                "Class": list(idx_to_class.values()),
                "Probability": [round(float(p) * 100, 2) for p in probs]
            })
            st.dataframe(df)
            st.bar_chart(df.set_index("Class"))
