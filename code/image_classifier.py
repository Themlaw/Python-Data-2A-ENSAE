import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        f.read(4)  # Magic number
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, num_rows, num_cols, 1)
    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        f.read(4)  # Magic number
        num_labels = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def load_data(data_dir):
    train_images_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    test_images_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')

    x_train = load_mnist_images(train_images_path)
    y_train = load_mnist_labels(train_labels_path)
    x_test = load_mnist_images(test_images_path)
    y_test = load_mnist_labels(test_labels_path)

    return (x_train, y_train), (x_test, y_test)

