import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ===============================
# KONFIGURASI HALAMAN (Wajib paling atas)
# ===============================
st.set_page_config(
    page_title="Deteksi Nominal Uang Rupiah",
    page_icon="💸",
    layout="centered"
)

# ===============================
# CSS INJECTION UTK TAMPILAN PROFESIONAL & SOFT
# ===============================
custom_css = """
<style>
    /* Import Font Modern dari Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

    /* Terapkan font ke seluruh elemen */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Sembunyikan elemen bawaan Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Background Aplikasi: Gradient biru sangat lembut */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%);
    }

    /* Ubah kontainer utama menjadi bentuk 'Card' melayang */
    .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 3rem !important;
        border-radius: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
        margin-top: 3rem;
        margin-bottom: 3rem;
        max-width: 650px !important;
        backdrop-filter: blur(10px);
    }

    /* Kustomisasi Area Upload File */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #cbd5e1 !important;
        background-color: #f8fafc !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        transition: all 0.3s ease;
    }
    
    /* Efek saat area upload di-hover */
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: #f1f5f9 !important;
        border-color: #94a3b8 !important;
    }

    /* Warna teks dan header */
    h1, h2, h3, h4 {
        color: #1e293b !important;
    }
    p {
        color: #475569 !important;
    }

    /* Kustomisasi Progress Bar agar warnanya senada */
    .stProgress > div > div > div > div {
        background-color: #3b82f6 !important;
        border-radius: 10px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ===============================
# LOAD MODEL DAN NAMA KELAS (Pakai Cache)
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
        return "Rp " + f"{angka:,}".replace(",", ".")
    except:
        return nominal

# ===============================
# TAMPILAN WEB (UI)
# ===============================
# Header Web
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2 style='font-weight: 600; margin-bottom: 0;'>💸 Deteksi Nominal Uang</h2>
        <p style='font-size: 0.95rem; margin-top: 5px;'>Unggah gambar uang kertas rupiah dan biarkan AI mengenali nominalnya dengan cepat.</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Pilih gambar uang",
    type=["jpg", "jpeg", "png"],
    label_visibility="hidden"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Menampilkan gambar dengan padding dan shadow agar rapi
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    with col2:
        st.image(image, use_container_width=True, clamp=True)
        st.markdown("<p style='text-align:center; font-size:0.8rem; color:#94a3b8;'>Gambar yang sedang diproses</p>", unsafe_allow_html=True)

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
    st.markdown("<hr style='border-top: 1px dashed #cbd5e1; margin: 2rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; font-weight: 500; margin-bottom: 1rem;'>📊 Hasil Analisis</h4>", unsafe_allow_html=True)

    # Kotak hasil prediksi yang lebih menonjol
    if confidence < 70:
        st.warning(f"⚠️ **Prediksi Sementara:** {format_rupiah(predicted_class)}", icon=None)
    else:
        st.success(f"✅ **Nominal Terdeteksi:** {format_rupiah(predicted_class)}", icon=None)

    # Indikator Confidence
    st.markdown(f"<p style='margin-bottom: 5px; font-size: 0.9rem; font-weight: 500;'>Tingkat Keyakinan: {confidence:.2f}%</p>", unsafe_allow_html=True)
    st.progress(int(confidence))

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detail Prediksi
    with st.expander("🔍 Lihat Detail Probabilitas Kelas Lainnya"):
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        for i in top_indices:
            nama_kelas = class_names[i]
            persen = predictions[0][i] * 100
            st.markdown(f"""
                <div style='display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #f1f5f9;'>
                    <span style='color: #475569;'>{format_rupiah(nama_kelas)}</span>
                    <span style='font-weight: 600; color: #1e293b;'>{persen:.2f}%</span>
                </div>
            """, unsafe_allow_html=True)