import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ===============================
# LOAD MODEL DAN NAMA KELAS
# ===============================
model = tf.keras.models.load_model("model_uang.keras")

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ===============================
# FORMAT RUPIAH
# ===============================
def format_rupiah(nominal):
    nominal = nominal.replace("Rpidr_", "")
    nominal = nominal.replace("idr_", "")
    nominal = nominal.replace("IDR_", "")
    nominal = nominal.replace("Rp", "")
    nominal = nominal.replace("rp", "")
    nominal = nominal.strip()

    try:
        angka = int(nominal)
        return "Rp" + f"{angka:,}".replace(",", ".")
    except:
        return nominal

# ===============================
# TAMPILAN WEB
# ===============================
st.set_page_config(
    page_title="Deteksi Nominal Uang Rupiah",
    page_icon="💵",
    layout="centered"
)

st.title("💵 Sistem Pengenalan Nominal Uang Kertas Rupiah")
st.write("Upload gambar uang kertas Rupiah, lalu sistem akan memprediksi nominalnya.")

uploaded_file = st.file_uploader(
    "Upload gambar uang",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Gambar Uang yang Diupload", use_container_width=True)

    # ===============================
    # PREPROCESSING GAMBAR
    # ===============================
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # ===============================
    # PREDIKSI
    # ===============================
    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = 100 * np.max(predictions[0])

    # ===============================
    # HASIL PREDIKSI
    # ===============================
    st.subheader("Hasil Prediksi")

    if confidence < 70:
        st.warning(f"Model belum terlalu yakin. Prediksi sementara: {format_rupiah(predicted_class)}")
    else:
        st.success(f"Nominal uang terdeteksi: {format_rupiah(predicted_class)}")

    st.write(f"Tingkat keyakinan: {confidence:.2f}%")

    st.write("Kemungkinan lain:")

    top_indices = np.argsort(predictions[0])[-3:][::-1]

    for i in top_indices:
        nama_kelas = class_names[i]
        persen = predictions[0][i] * 100
        st.write(f"- {format_rupiah(nama_kelas)}: {persen:.2f}%")