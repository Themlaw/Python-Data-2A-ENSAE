import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from MNIST import MnistDataloader

def preprocess_data(x_train, y_train, x_test, y_test):
    """Preprocesses the test data by normalizing and one-hot encoding the labels.

    :param x_train: training images
    :param y_train: training labels
    :param x_test: test images
    :param y_test: test labels
    :returns: preprocessed training and test data
    """
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def build_model():
    """Builds and compiles a simple CNN model.

    :returns: a compiled Keras model
    """
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_evaluate(data_dir='data/MNIST', model_save_path='data/models/mnist_cnn.keras'):
    """Trains and evaluates the CNN model.

    :param data_dir: The directory where the MNIST data is stored.
    :param model_save_path: The path to save the trained model.
    :raises Exception: if data loading or model training fails
    """
    # Charger les données via MnistDataloader
    mnist_loader = MnistDataloader(data_dir=data_dir)
    try:
        (x_train, y_train), (x_test, y_test) = mnist_loader.load_data()
    except Exception as e:
        mnist_loader.download_mnist()
        (x_train, y_train), (x_test, y_test) = mnist_loader.load_data()
    
    # Convertir en arrays numpy et ajouter dimension pour CNN (28, 28) -> (28, 28, 1)
    x_train = np.array([np.array(img).reshape(28, 28, 1) for img in x_train])
    x_test = np.array([np.array(img).reshape(28, 28, 1) for img in x_test])
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    (x_train, y_train), (x_test, y_test) = preprocess_data(x_train, y_train, x_test, y_test)

    model = build_model()
    model.summary()

    print("\nTraining model...")
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'\nTest accuracy: {test_acc}')

    # Créer le dossier data/models s'il n'existe pas
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to : {model_save_path}")

if __name__ == '__main__':
    train_and_evaluate()
