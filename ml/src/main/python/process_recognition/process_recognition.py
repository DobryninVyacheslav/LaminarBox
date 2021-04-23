import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.preprocessing.image_preprocessing import HORIZONTAL_AND_VERTICAL

from ml.src.main.python.utils import tf_utils

# Download and explore the dataset
train_and_val_data_dir = pathlib.Path("ml/src/resources/process_recognition_data/process_photo/train_and_val")
test_data_dir = pathlib.Path("ml/src/resources/process_recognition_data/process_photo/test")
print("Total train and val photo number: ", len(list(train_and_val_data_dir.glob('*/*.jpg'))))
print("Total test photo number: ", len(list(test_data_dir.glob('*/*.jpg'))))

# Create datasets
batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_and_val_data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_and_val_data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_data_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(train_ds.class_names)
print("Classes: ", class_names)

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation
augmentation_and_rescale = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip(HORIZONTAL_AND_VERTICAL,
                                                     input_shape=(img_height,
                                                                  img_width,
                                                                  3)),
        layers.experimental.preprocessing.RandomRotation(0.05),
        layers.experimental.preprocessing.RandomZoom(0.05),
        layers.experimental.preprocessing.RandomContrast(0.1),
        layers.experimental.preprocessing.Rescaling(1. / 255),
    ]
)

aug_train_ds = train_ds.map(lambda x, y: (augmentation_and_rescale(x, training=True), y))

# Visualize the data
tf_utils.show_images(tf_utils.normalize_ds(train_ds), class_names=class_names)
tf_utils.show_images(aug_train_ds, class_names=class_names)

# Create the model
model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
epochs = 20
history = model.fit(
    aug_train_ds.concatenate(tf_utils.normalize_ds(train_ds)),
    validation_data=val_ds,
    epochs=epochs
)

tf_utils.show_training_results(history, epochs)

# Predict on new data
check_photo = test_ds.take(1)
predictions = model.predict(check_photo)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
)

plt.figure(figsize=(10, 10))
for images, _ in check_photo:
    plt.imshow(images[0].numpy().astype("uint8"))

plt.show()

loss, acc = model.evaluate(test_ds)
print("Accuracy: ", acc)

# Save model to json format
tfjs.converters.save_keras_model(model, "src/main/resources/static/model/process")
