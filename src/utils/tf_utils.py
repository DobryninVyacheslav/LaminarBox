import matplotlib.pyplot as plt


def get_images(ds):
    img, labs = next(iter(ds))
    return img


def print_size(ds):
    print("Size: ", len(get_images(ds)))


def show_plt(ds, subplot_number=9, subplot_x_num=3, subplot_y_num=3):
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(subplot_number):
            ax = plt.subplot(subplot_x_num, subplot_y_num, i + 1)
            plt.imshow(images[i])
            plt.axis("off")
    plt.show()
