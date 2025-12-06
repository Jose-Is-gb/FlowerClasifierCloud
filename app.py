import os
import numpy as np
import tensorflow as tf
#import tensorflow_datasets as tfds#
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, render_template

app = Flask(__name__)

# -----------------------------
# CARGAR MODELO
# -----------------------------
model = tf.keras.models.load_model("model.h5")
print("Modelo cargado correctamente.")

# -----------------------------
# CARGAR LABELS
# -----------------------------
with open("labels.txt", "r", encoding="utf-8") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

IMG_SIZE = 224

# -----------------------------
# DIRECTORIO DE IMÁGENES
# -----------------------------
SAMPLES_DIR = "static/samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)

# -----------------------------
# DESCARGAR DATASET DE VALIDACIÓN
# -----------------------------
#print("Descargando Oxford Flowers 102...")

#(ds_train, ds_val), ds_info = tfds.load(
 #   "oxford_flowers102",
  #  split=["train", "validation"],
   # as_supervised=True,
    #with_info=True
#)

# -----------------------------
# GUARDAR 50 IMÁGENES CON NOMBRES CORRECTOS
# -----------------------------
#if len(os.listdir(SAMPLES_DIR)) == 0:
#    print("Guardando 50 imágenes con su etiqueta real...")
#    count = 0
#    for img, label in ds_val.take(50):
#       real_class = int(label.numpy())  # clase correcta 0–101
#        path = f"{SAMPLES_DIR}/{count}_{real_class}.jpg"
#       tf.keras.preprocessing.image.save_img(path, img.numpy())
#        count += 1
#    print("Imágenes guardadas correctamente.")


# -----------------------------
# EXTRAER LABEL REAL DEL NOMBRE
# archivo ejemplo → "12_54.jpg"
# -----------------------------
def obtener_label_real_desde_nombre(filename):
    if "_" not in filename:
        return "desconocido"    

    try:
        etiqueta = int(filename.split("_")[1].split(".")[0])
        return CLASS_NAMES[etiqueta]
    except:
        return "desconocido"


# -----------------------------
# FUNCIÓN DE PREDICCIÓN
# -----------------------------
def predecir(img_path):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])

    return CLASS_NAMES[idx], conf


# -----------------------------
# RUTA PRINCIPAL
# -----------------------------
@app.route("/")
def index():

    files = [
        f for f in os.listdir(SAMPLES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    data = []

    for file in files:
        img_path = f"{SAMPLES_DIR}/{file}"

        # predicción
        pred_class, conf = predecir(img_path)

        # etiqueta real
        real_label = obtener_label_real_desde_nombre(file)

        data.append({
            "ruta": img_path,
            "pred": f"{pred_class} ({conf:.4f})",
            "real": real_label,
            "correcta": (pred_class == real_label)
        })

    return render_template("index.html", data=data)


# -----------------------------
# EJECUTAR SERVIDOR
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)