import tensorflow as tf
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import kagglehub

path = kagglehub.dataset_download(
    "abdallahalidev/plantvillage-dataset"
)

print(path)

dataset_path = f"{path}/plantvillage dataset/color"
print(dataset_path)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=32,
)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=32,
)
class_names = train_dataset.class_names
print("Class names:", class_names)

# ______________________train dataset________________________
train_dataset = train_dataset.map(
    lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
validation_dataset = validation_dataset.map(
    lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
"You can add a cache() function, which greatly increases training speed, but the downside is that it will consume RAM."
"I don't recommend training your model with it enabled if you have less than 64 GB  of RAM, but if you have more than 64 GB of RAM, it can significantly speed up training."
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
# ___________________________Loading_the_model________________________
model = tf.keras.models.load_model(r"/content/plant_disease_model.keras")
model.summary()
base_model = model.layers[1]
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False
#_______________________compile the model________________________
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
# ________________________training__________________________
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "/content/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
history_fine = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=100,
    callbacks=[early_stop]
)
model_path = os.path.abspath("plant_disease_model.keras")
model.save(model_path)
print("Model saved at:")
print(model_path)
from google.colab import files
files.download("/content/plant_disease_model.keras")
