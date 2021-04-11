import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


def get_images(ds, as_numpy_array=False, with_normalize=False):
    if with_normalize:
        ds = normalize_ds(ds)
    img, labs = next(iter(ds))
    if as_numpy_array:
        return img.numpy().astype("uint8")
    else:
        return img


def print_size(ds):
    print("Size: ", len(get_images(ds)))


def show_images(ds, subplot_number=9, subplot_x_num=3,
                subplot_y_num=3, with_class_name=False, with_numpy_as_type=True):
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(subplot_number):
            ax = plt.subplot(subplot_x_num, subplot_y_num, i + 1)
            if with_numpy_as_type:
                plt.imshow(images[i].numpy().astype("uint8"))
            else:
                plt.imshow(images[i])
            if with_class_name:
                plt.title(ds.class_names[labels[i]])
            plt.axis("off")
    plt.show()


def show_training_results(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def normalize_ds(ds):
    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
    return ds.map(lambda x, y: (normalization_layer(x), y))


def augment_data(data_dir, height=180, width=180, show_plt=False):
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                     shear_range=0.2,
                                                                     zoom_range=0.2,
                                                                     vertical_flip=True,
                                                                     horizontal_flip=True)

    data = data_generator.flow_from_directory(data_dir, target_size=(height, width))

    if show_plt:
        batch = data[0]
        plt.figure(figsize=(10, 10))
        for i in range(32):
            ax = plt.subplot(4, 8, i + 1)
            plt.imshow(batch[0][i])
            plt.axis("off")
        plt.show()

    return data
