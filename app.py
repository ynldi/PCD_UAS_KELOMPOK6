import os
import json
import zipfile
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

# ======================
# PATCH KOMPATIBILITAS MODEL .H5
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

    # Load classes.json (opsional)
    classes_path = "classes.json"
    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            data = json.load(f)
        idx_to_class = {int(k): v for k, v in data.items()}
    else:
        # Tambahkan label dengan bahasa Indonesia
        idx_to_class = {
    0: 'apple (apel)', 1: 'banana (pisang)', 2: 'beetroot (bit)', 3: 'bell pepper (paprika)', 4: 'cabbage (kubis)',
    5: 'capsicum (cabai hijau)', 6: 'carrot (wortel)', 7: 'cauliflower (kembang kol)', 8: 'chilli pepper (cabai)', 9: 'corn (jagung)',
    10: 'cucumber (mentimun)', 11: 'eggplant (terong)', 12: 'garlic (bawang putih)', 13: 'ginger (jahe)', 14: 'grapes (anggur)',
    15: 'jalepeno (jalapeno)', 16: 'kiwi (kiwi)', 17: 'lemon (lemon)', 18: 'lettuce (selada)', 19: 'mango (mangga)',
    20: 'onion (bawang merah)', 21: 'orange (jeruk)', 22: 'paprika (paprika)', 23: 'pear (pir)', 24: 'peas (kacang polong)',
    25: 'pineapple (nanas)', 26: 'pomegranate (delima)', 27: 'potato (kentang)', 28: 'raddish (lobak)',
    29: 'soy beans (kedelai)', 30: 'spinach (bayam)', 31: 'sweetcorn (jagung manis)', 32: 'sweetpotato (ubi jalar)',
    33: 'tomato (tomat)', 34: 'turnip (lobak putih)', 35: 'watermelon (semangka)', 
    36: 'strawberry (stroberi)'   # ✅ tambahan kelas ke-36
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

st.title("🥕 Fruits and Vegetables Classification")

st.sidebar.title("🔧 Pengaturan & Input")
menu_choice = st.sidebar.radio("Pilih metode input:", ["Upload Gambar", "Kamera", "📊 Data Prediksi"])
show_prob = st.sidebar.checkbox("Tampilkan tabel probabilitas", True)
st.sidebar.markdown("---")
st.sidebar.write(f"Model input size: **{IMG_SIZE[0]}×{IMG_SIZE[1]}**")

# --- Menu Upload File ---
if menu_choice == "Upload Gambar":
    st.header("📂 Upload Gambar")
    uploaded = st.file_uploader("Upload gambar buah/sayur:", type=["jpg", "jpeg", "png"])

    if uploaded:
        try:
            img = Image.open(uploaded)
            st.image(img, caption="Gambar diupload", use_container_width=True)

            if st.button("🔍 Prediksi dari Upload"):
                with st.spinner("Memproses..."):
                    label, probs = predict(img)
                    conf = float(np.max(probs) * 100)

                st.subheader("Hasil Prediksi (Upload)")
                st.write(f"**Label:** `{label}`")
                st.write(f"**Confidence:** `{conf:.2f}%`")

                add_to_history("Upload", label, conf)

                if show_prob:
                    df = pd.DataFrame({
                        "Class": list(idx_to_class.values()),
                        "Probability": [round(float(p) * 100, 2) for p in probs]
                    })
                    st.dataframe(df)
                    st.bar_chart(df.set_index("Class"))
        except Exception:
            st.error("File yang diupload tidak bisa dibaca sebagai gambar. Pastikan format JPG/PNG.")

# --- Menu Kamera ---
elif menu_choice == "Kamera":
    st.header("📸 Ambil Foto dengan Kamera")
    camera_file = st.camera_input("Klik 'Open Camera' lalu ambil gambar")

    if camera_file:
        try:
            img = Image.open(camera_file)
            st.image(img, caption="Foto dari kamera", use_container_width=True)

            if st.button("🔍 Prediksi dari Kamera"):
                with st.spinner("Memproses..."):
                    label, probs = predict(img)
                    conf = float(np.max(probs) * 100)

                st.subheader("Hasil Prediksi (Kamera)")
                st.write(f"**Label:** `{label}`")
                st.write(f"**Confidence:** `{conf:.2f}%`")

                add_to_history("Kamera", label, conf)

                if show_prob:
                    df = pd.DataFrame({
                        "Class": list(idx_to_class.values()),
                        "Probability": [round(float(p) * 100, 2) for p in probs]
                    })
                    st.dataframe(df)
                    st.bar_chart(df.set_index("Class"))
        except Exception:
            st.error("Foto dari kamera tidak bisa dibaca. Silakan coba lagi.")

# --- Menu Data

# --- Menu Data Prediksi ---
elif menu_choice == "📊 Data Prediksi":
    st.header("📊 Data Prediksi")

    if st.session_state["history"]:
        df_hist = pd.DataFrame(st.session_state["history"])
        st.dataframe(df_hist)

        if st.button("🗑️ Hapus Semua Data Prediksi"):
            clear_history()
            st.success("Data prediksi berhasil dihapus.")
    else:
        st.info("Belum ada data prediksi yang tersimpan.")


