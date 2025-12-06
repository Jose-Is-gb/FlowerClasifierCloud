import os
import numpy as np
import tensorflow as tf
# Necesario para el runtime de TFLite
from tensorflow.lite.python.interpreter import Interpreter 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, render_template

app = Flask(__name__)

# -----------------------------
# ARCHIVOS DEL MODELO LITE
# -----------------------------
MODEL_PATH = "model_quantized.tflite"
IMG_SIZE = 224

# -----------------------------
# CARGAR MODELO TFLITE
# -----------------------------
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()


print("Iniciando aplicación Flask.")
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

# Obtener los tensores de entrada y salida para la inferencia
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Modelo TFLite cargado y tensores asignados correctamente.")

# -----------------------------
# CARGAR LABELS
# -----------------------------
with open("labels.txt", "r", encoding="utf-8") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

# -----------------------------
# DIRECTORIO DE IMÁGENES (el resto del código es igual)
# -----------------------------
SAMPLES_DIR = "static/samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)

# ... (Las funciones obtener_label_real_desde_nombre van aquí)

# -----------------------------
# FUNCIÓN DE PREDICCIÓN (MODIFICADA para TFLite)
# -----------------------------
def predecir(img_path):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    # TFLite generalmente requiere float32 para la inferencia cuantificada
    arr = img_to_array(img).astype(np.float32) 
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    # 1. Colocar el tensor de entrada
    interpreter.set_tensor(input_details[0]['index'], arr)
    
    # 2. Ejecutar la inferencia
    interpreter.invoke()
    
    # 3. Obtener el resultado
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    
    idx = int(np.argmax(preds))
    conf = float(preds[idx])

    return CLASS_NAMES[idx], conf

# -----------------------------
# RUTA PRINCIPAL (Sin Cambios)
# -----------------------------
@app.route("/")
def index():
    # ... (El resto de tu lógica de rutas aquí)

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
# EJECUTAR SERVIDOR (Sin Cambios)
# -----------------------------
# if __name__ == "__main__":
#    port = int(os.environ.get("PORT", 5000))
#    app.run(host="0.0.0.0", port=port)

# **Nota:** Eliminamos el bloque '__main__' para evitar conflictos con Gunicorn.