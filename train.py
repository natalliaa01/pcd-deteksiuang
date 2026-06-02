import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ===============================
# KONFIGURASI DATASET
# ===============================
dataset_path = "dataset"
img_size = (224, 224)
batch_size = 16

# ===============================
# LOAD DATASET
# ===============================
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Kelas nominal uang:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ===============================
# DATA AUGMENTATION
# ===============================
# Tidak pakai RandomFlip karena uang kalau dibalik bisa bikin model bingung
data_augmentation = models.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# ===============================
# BASE MODEL MOBILENETV2
# ===============================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# ===============================
# MODEL
# ===============================
model = models.Sequential([
    data_augmentation,

    # Pengganti preprocess_input.
    # MobileNetV2 butuh input -1 sampai 1.
    layers.Rescaling(1./127.5, offset=-1),

    base_model,

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax")
])

# ===============================
# COMPILE MODEL
# ===============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# TRAINING
# ===============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

# ===============================
# SIMPAN MODEL
# ===============================
model.save("model_uang.keras")

with open("class_names.txt", "w") as f:
    for class_name in class_names:
        f.write(class_name + "\n")

print("Model berhasil disimpan sebagai model_uang.keras")
print("Nama kelas berhasil disimpan sebagai class_names.txt")

# ===============================
# GRAFIK AKURASI
# ===============================
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Grafik Akurasi Model")
plt.show()