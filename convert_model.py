import tensorflow as tf
import os

MODEL_H5_PATH = "model.h5"
TFLITE_MODEL_PATH = "model_quantized.tflite"

print(f"Tamaño original: {os.path.getsize(MODEL_H5_PATH) / (1024*1024):.2f} MB")

# Cargar el modelo original
model = tf.keras.models.load_model(MODEL_H5_PATH)

# Crear el convertidor
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 1. Configurar para Cuantificación (tamaño reducido)
# Esto reduce el modelo de 32 bits a 16 bits (float16), reduciendo a la mitad su tamaño y RAM
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16] 

# 2. Convertir y guardar
tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"Modelo TFLite guardado en: {TFLITE_MODEL_PATH}")
print(f"Nuevo tamaño: {os.path.getsize(TFLITE_MODEL_PATH) / (1024*1024):.2f} MB")