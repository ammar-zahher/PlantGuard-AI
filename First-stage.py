# import required libraries
from keras.models import Sequential
from keras.layers import Dense, Flatten
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import os

# ____________________________________________________________________architecture of the model___________________________________________________________________________
""
# __________________________arrange data________________________
dataset_path = r"C:\Users\hesha\.cache\kagglehub\datasets\abdallahalidev\plantvillage-dataset\versions\3\plantvillage dataset\color"
print(os.listdir(dataset_path))
# if your tensorflow version is 2.9 or higher, you can use the following code to load the dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    # we set the seed to ensure that the same images are used for training and validation across different runs
    seed=42,
    image_size=(224, 224),
    batch_size=32,
)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    # we set the seed to ensure that the same images are used for training and validation across different runs
    seed=42,
    image_size=(224, 224),
    batch_size=32,
)
class_names = train_dataset.class_names

print("Class names:", class_names)
# ----------------data_augmentation--------------------------
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ]
)
# __________________________Improving performance________________________
train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y))

validation_dataset = validation_dataset.map(lambda x, y: (preprocess_input(x), y))

# good addition to improve performance by using cache and prefetch
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(AUTOTUNE)
# __________________________________________________
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
for layer in base_model.layers:
    layer.trainable = False
# ________________________training__________________________

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)
model = Sequential(
    [
        data_augmentation,
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(len(class_names), activation="softmax"),
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()
history = model.fit(
    train_dataset, validation_data=validation_dataset, epochs=20, callbacks=[early_stop]
)
model_path = os.path.abspath("plant_disease_model.keras")

model.save(model_path)

print("Model saved at:")
print(model_path)

import matplotlib.pyplot as plt
#---------------------------------------------good addition to plot the training and validation accuracy and loss----------------------------------------------
# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
