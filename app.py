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

try:
    tf.keras.utils.set_keras_version("2")
except:
    pass

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"

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
# CONFIG HALAMAN
# =========================
st.set_page_config(
    page_title="Fruits & Vegetables Classifier",
    page_icon="🥕",
    layout="centered"
)

# =========================
# EKSTRAK MODEL ZIP
# =========================
def extract_model_zip():
    zip_path = "cnn_model.zip"
    extract_to = "./"
    if os.path.exists("cnn_model.h5"):
        return
    if not os.path.exists(zip_path):
        st.error("File ZIP tidak ditemukan.")
        st.stop()
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        st.success("📦 Model berhasil diekstrak.")
    except Exception as e:
        st.error(f"Gagal ekstrak ZIP: {e}")
        st.stop()

# =========================
# LOAD MODEL & KELAS
# =========================
@st.cache_resource
def load_model_and_classes():
    extract_model_zip()
    model_path = "cnn_model.h5"
    if not os.path.exists(model_path):
        st.error("cnn_model.h5 tidak ditemukan setelah extract ZIP.")
        st.stop()
    try:
        model = load_model(model_path, compile=False)
    except Exception:
        model = load_model(model_path, compile=False, custom_objects=CUSTOM_OBJECTS)

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
# PREPROCESS & PREDIKSI
# =========================
def preprocess_image(image, img_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    right = left + min_side
    bottom = top + min_side
    image = image.crop((left, top, right, bottom))
    image = image.resize(img_size)
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
# SESSION STATE UNTUK HISTORY
# =========================
if "history" not in st.session_state:
    st.session_state["history"] = []

def add_to_history(source, label, conf):
    st.session_state["history"].append({
        "source": source,
        "label": label,
        "confidence": f"{conf:.2f}%"
    })

def clear_history():
    st.session_state["history"] = []

# =========================
# UI STREAMLIT
# =========================
if os.path.exists("banner.png"):
    st.image("banner.png", use_container_width=True)

# Pilihan bahasa
lang = st.sidebar.radio("Bahasa / Language:", ["Indonesia", "English"])

title_text = "🥕 Klasifikasi Buah & Sayur" if lang == "Indonesia" else "🥕 Fruits and Vegetables Classification"
st.title(title_text)

st.sidebar.title("🔧 Pengaturan & Input")
menu_choice = st.sidebar.radio("Pilih metode input:" if lang == "Indonesia" else "Choose input method:",
                               ["Upload Gambar" if lang == "Indonesia" else "Upload Image",
                                "Kamera" if lang == "Indonesia" else "Camera",
                                "📊 Data Prediksi" if lang == "Indonesia" else "📊 Prediction Data"])

show_prob = st.sidebar.checkbox("Tampilkan tabel probabilitas" if lang == "Indonesia" else "Show probability table", True)
st.sidebar.markdown("---")
st.sidebar.write(f"Ukuran input model: **{IMG_SIZE[0]}×{IMG_SIZE[1]}**" if lang == "Indonesia" else f"Model input size: **{IMG_SIZE[0]}×{IMG_SIZE[1]}**")

# --- Menu Upload File ---
if menu_choice.startswith("Upload"):
    st.header("📂 Upload Gambar" if lang == "Indonesia" else "📂 Upload Image")
    uploaded = st.file_uploader("Upload gambar buah/sayur:" if lang == "Indonesia" else "Upload fruit/vegetable image:",
                                type=["jpg", "jpeg", "png"])
    if uploaded:
        try:
            img = Image.open(uploaded)
            st.image(img, caption="Gambar diupload" if lang == "Indonesia" else "Image uploaded", use_container_width=True)
            if st.button("🔍 Prediksi dari Upload" if lang == "Indonesia" else "🔍 Predict from Upload"):
                with st.spinner("Memproses..." if lang == "Indonesia" else "Processing..."):
                    label, probs = predict(img)
                    conf = float(np.max(probs) * 100)
                st.subheader("Hasil Prediksi (Upload)" if lang == "Indonesia" else "Prediction Result (Upload)")
                st.write(f"**Label:** `{label}`")
                st.write(f"**Kepercayaan:** `{conf:.2f}%`" if lang == "Indonesia" else f"**Confidence:** `{conf:.2f}%`")
                add_to_history("Upload", label, conf)
                if show_prob:
                    df = pd.DataFrame({
                        "Class": list(idx_to_class.values()),
                        "Probability": [round(float(p) * 100, 2) for p in probs]
                    })
                    st.dataframe(df)
                    st.bar_chart(df.set_index("Class"))
        except Exception:
            st.error("File yang diupload tidak bisa dibaca sebagai gambar. Pastikan format JPG/PNG." if lang == "Indonesia" else "Uploaded file is not a valid image. Please use JPG/PNG.")

# --- Menu Kamera ---
elif menu_choice.startswith("Kamera") or menu_choice.startswith("Camera"):
    st.header("📸 Ambil Foto dengan Kamera" if lang == "Indonesia" else "📸 Take Photo with Camera")
    camera_file = st.camera_input("Klik 'Open Camera' lalu ambil gambar" if lang == "Indonesia" else "Click 'Open Camera' then take a photo")
    if camera_file:
        try:
            img = Image.open(camera_file)
            st.image(img, caption="Foto dari kamera" if lang == "Indonesia" else "Photo from camera", use_container_width=True)
            if st.button("🔍 Prediksi dari Kamera" if lang == "Indonesia" else "🔍 Predict from Camera"):
                with
