import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ===============================
# KONFIGURASI HALAMAN (Wajib paling atas)
# ===============================
st.set_page_config(
    page_title="Deteksi Nominal Uang Rupiah",
    page_icon="💵",
    layout="centered"
)

# ===============================
# HILANGKAN ELEMEN BAWAAN STREAMLIT (Tampilan Minimalis)
# ===============================
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;} /* Sembunyikan menu hamburger */
            footer {visibility: hidden;}    /* Sembunyikan footer 'Made with Streamlit' */
            header {visibility: hidden;}    /* Sembunyikan header/garis atas */
            
            /* Bikin background abu-abu sangat muda dan font lebih bersih */
            .stApp {
                background-color: #f8fafc; 
            }
            
            /* Sembunyikan batas padding atas yang terlalu lebar */
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ===============================
# LOAD MODEL DAN NAMA KELAS (Menggunakan Cache agar cepat)
# ===============================
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("model_uang.keras")
    with open("class_names.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = load_model_and_classes()

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
# TAMPILAN WEB (UI)
# ===============================
# Judul menggunakan markdown agar bisa diatur posisinya di tengah
st.markdown("<h2 style='text-align: center; color: #1e293b; margin-bottom: 0px;'>💵 Deteksi Nominal Uang Rupiah</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b; margin-bottom: 30px;'>Unggah gambar uang kertas untuk memulai pengenalan nominal</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Pilih gambar uang",
    type=["jpg", "jpeg", "png"],
    label_visibility="hidden" # Sembunyikan label bawaan agar lebih bersih
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Menampilkan gambar dengan sudut membulat menggunakan kolom agar rapi
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)

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
    st.markdown("---")
    st.markdown("<h4 style='text-align: center; color: #334155;'>Hasil Analisis</h4>", unsafe_allow_html=True)

    if confidence < 70:
        st.warning(f"⚠️ Model belum terlalu yakin. Prediksi sementara: **{format_rupiah(predicted_class)}**")
    else:
        st.success(f"✅ Nominal uang terdeteksi: **{format_rupiah(predicted_class)}**")

    # Progress bar untuk tingkat keyakinan (confidence) agar visualnya lebih menarik
    st.write(f"**Tingkat Keyakinan (Confidence):** {confidence:.2f}%")
    st.progress(int(confidence))

    with st.expander("Lihat kemungkinan lainnya (Detail Prediksi)"):
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        for i in top_indices:
            nama_kelas = class_names[i]
            persen = predictions[0][i] * 100
            st.write(f"- {format_rupiah(nama_kelas)}: **{persen:.2f}%**")