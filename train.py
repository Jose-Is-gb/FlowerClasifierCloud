# train.py
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os

# SETTINGS
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 8
EPOCHS_FINE = 8
NUM_SAMPLES_TO_SAVE = 50  # opcional, si guardas ejemplos

# 1) Load dataset
print("Cargando Oxford Flowers 102...")
(ds_train, ds_val), ds_info = tfds.load(
    "oxford_flowers102",
    split=["train", "validation"],
    as_supervised=True,
    with_info=True
)
CLASS_NAMES = ds_info.features["label"].names
num_classes = len(CLASS_NAMES)
print("Clases:", num_classes)

# save labels
with open("labels.txt", "w", encoding="utf-8") as f:
    for n in CLASS_NAMES:
        f.write(n + "\n")

# 2) Data augmentation + preprocess function
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.12),
    layers.RandomContrast(0.08),
])

def preprocess(image, label, augment=False):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    if augment:
        image = data_augmentation(image)
    # MobileNetV2 preprocess_input expects scale to [-1,1]
    image = preprocess_input(image)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE

train_ds = ds_train.map(lambda x, y: preprocess(x, y, augment=True)).shuffle(2000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds   = ds_val.map(lambda x, y: preprocess(x, y, augment=False)).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# 3) Build model
base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 4) Callbacks
os.makedirs("checkpoints", exist_ok=True)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "checkpoints/best_head.h5", save_best_only=True, monitor="val_loss"
)
reduce_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)
early_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

# 5) Train head
print("Entrenando cabeza (head) ...")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=[checkpoint_cb, reduce_cb, early_cb]
)

# load best head weights
if os.path.exists("checkpoints/best_head.h5"):
    model.load_weights("checkpoints/best_head.h5")

# 6) Fine-tuning: unfreeze top layers of base_model
print("Fine-tuning: desbloqueando parte de MobileNetV2 ...")
base_model.trainable = True

# freeze early layers, unfreeze last blocks
for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_cb2 = tf.keras.callbacks.ModelCheckpoint(
    "checkpoints/best_finetuned.h5", save_best_only=True, monitor="val_loss"
)
reduce_cb2 = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7)
early_cb2 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=[checkpoint_cb2, reduce_cb2, early_cb2]
)

# 7) Save final model
print("Guardando modelo final model.h5 ...")
model.save("model.h5")
print("Entrenamiento terminado. model.h5 y labels.txt listos.")
