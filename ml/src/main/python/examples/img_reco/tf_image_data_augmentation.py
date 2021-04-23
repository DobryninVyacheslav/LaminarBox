import pathlib
from utils import tf_utils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

data_dir = pathlib.Path(
    "/home/slava/Common/PycharmProjects/LaminarBox/resources/process_recognition_data")

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names

image, label = next(iter(train_ds))


# _ = plt.imshow(image[0].numpy().astype("uint8"))
# _ = plt.title(class_names[label[0]])
# plt.show()


def visualize(original, augmented, with_colorbar=False):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original[0].numpy().astype("uint8"))

    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(augmented[0].numpy().astype("uint8"))
    if with_colorbar:
        plt.colorbar()
    plt.show()


flipped = tf.image.flip_left_right(image)  # 1
# visualize(image, flipped)

grayscaled = tf.image.rgb_to_grayscale(image)  # 1
# visualize(image, tf.squeeze(grayscaled), True)

saturated = tf.image.adjust_saturation(image, 3)  # 1
# visualize(image, saturated)

bright = tf.image.adjust_brightness(image, 10)  # 1
# visualize(image, bright)

cropped = tf.image.central_crop(image, central_fraction=0.5)  # 1
# visualize(image, cropped)

rotated = tf.image.rot90(image)  # 1
# visualize(image, rotated)

# for i in range(3):  # 1
#     seed = (i, 0)  # tuple of size (2,)
#     stateless_random_brightness = tf.image.stateless_random_brightness(
#         image, max_delta=1.95, seed=seed)
#     visualize(image, stateless_random_brightness)
#
# for i in range(3):  # 1
#     seed = (i, 0)  # tuple of size (2,)
#     stateless_random_contrast = tf.image.stateless_random_contrast(
#         image, lower=0.1, upper=0.9, seed=seed)
#     visualize(image, stateless_random_contrast)
#
# for i in range(3): # 0.5
#     seed = (i, 0)  # tuple of size (2,)
#     stateless_random_crop = tf.image.stateless_random_crop(
#         image, size=[30, 180, 180, 3], seed=seed)
#     visualize(image, stateless_random_crop)

IMG_SIZE = 180


def resize_and_rescale(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = (image / 255.0)
    return image, label


def augment(image_label, seed):
    image, label = image_label
    image, label = resize_and_rescale(image, label)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
    # Make a new seed
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    # Random crop back to the original size
    image = tf.image.stateless_random_crop(
        image, size=[len(image), IMG_SIZE, IMG_SIZE, 3], seed=seed)
    # Random brightness
    image = tf.image.stateless_random_brightness(
        image, max_delta=0.5, seed=new_seed)
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.adjust_saturation(image, 3)
    image = tf.image.central_crop(image, central_fraction=0.5)
    image = tf.image.flip_left_right(image)
    return image, label


b_train_ds = train_ds
# print(tf_utils.get_images(train_ds))
tf_utils.print_size(train_ds)

# Create counter and zip together with train dataset
AUTOTUNE = tf.data.AUTOTUNE

rng = tf.random.Generator.from_seed(123, alg='philox')


# A wrapper function for updating seeds
def f(x, y):
    seed = rng.make_seeds(2)[0]
    image, label = augment((x, y), seed)
    return image, label


train_ds = (
    train_ds
        .shuffle(1000)
        .map(f, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
)


# image, label = next(iter(train_ds))
# _ = plt.imshow(image[0])
# plt.show()
