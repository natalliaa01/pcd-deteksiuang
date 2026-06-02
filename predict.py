import tensorflow as tf
import numpy as np
from PIL import Image

# ===============================
# LOAD MODEL DAN NAMA KELAS
# ===============================
model = tf.keras.models.load_model("model_uang.keras")

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

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
# INPUT GAMBAR
# ===============================
image_path = input("Masukkan path gambar uang: ")

# ===============================
# PREPROCESSING GAMBAR
# ===============================
img = Image.open(image_path).convert("RGB")
img = img.resize((224, 224))

img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0)

# ===============================
# PREDIKSI
# ===============================
predictions = model.predict(img_array)

predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]
confidence = 100 * np.max(predictions[0])

print("==============================")
print(f"Nominal uang terdeteksi: {format_rupiah(predicted_class)}")
print(f"Tingkat keyakinan: {confidence:.2f}%")
print("==============================")

print("Kemungkinan lain:")
top_indices = np.argsort(predictions[0])[-3:][::-1]

for i in top_indices:
    print(f"- {format_rupiah(class_names[i])}: {predictions[0][i] * 100:.2f}%")