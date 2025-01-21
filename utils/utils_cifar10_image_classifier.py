import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_cifar10():
    """Loads CIFAR-10 dataset and normalizes it."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
    return (x_train, y_train), (x_test, y_test)

def plot_images(images, labels, class_names):
    """Plots several images from the dataset."""
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i][0]])
        plt.axis('off')
    plt.show()
