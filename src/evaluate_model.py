import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from MNIST import MnistDataloader

def preprocess_data(x_test, y_test):
    """Preprocesses the test data by normalizing and one-hot encoding the labels.

    :param x_test: test images
    :param y_test: test labels
    :returns: preprocessed test images and labels
    """
    x_test = x_test.astype('float32') / 255.0
    y_test = to_categorical(y_test, 10)
    return x_test, y_test

def evaluate_existing_model(model_path='data/models/mnist_cnn.keras', data_dir='data/MNIST'):
    """Loads a pre-trained Keras model and evaluates it on the MNIST test set.

    :param model_path: The path to the saved Keras model file.
    :param data_dir: The directory where the MNIST data is stored.
    :raises FileNotFoundError: if the model or data is not found
    """
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading test data...")
    mnist_loader = MnistDataloader(data_dir=data_dir)
    try:
        (_, _), (x_test, y_test) = mnist_loader.load_data()
    except FileNotFoundError:
        print("Test data not found. Please ensure the MNIST dataset is available in the data/MNIST directory.")
        return
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return

    # Convert to numpy arrays and reshape for the CNN
    x_test = np.array([np.array(img).reshape(28, 28, 1) for img in x_test])
    y_test = np.array(y_test)

    # Preprocess the data
    x_test, y_test = preprocess_data(x_test, y_test)

    print("\nEvaluating model on the test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'\nTest Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    evaluate_existing_model()
