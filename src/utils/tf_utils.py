import matplotlib.pyplot as plt


def get_images(ds):
    img, labs = next(iter(ds))
    return img


def print_size(ds):
    print("Size: ", len(get_images(ds)))


def show_images(ds, subplot_number=9, subplot_x_num=3, subplot_y_num=3):
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(subplot_number):
            ax = plt.subplot(subplot_x_num, subplot_y_num, i + 1)
            plt.imshow(images[i])
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
