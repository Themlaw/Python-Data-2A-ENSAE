import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from MNIST import MnistDataloader
import matplotlib.pyplot as plt

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


def evaluate_vs_noise(model_path='data/models/mnist_cnn.keras', data_dir='data/MNIST',
                      noise_type='gaussian', epsilon=0.05, sample_size=None):
    """Evaluate a saved model across a range of noise levels and plot the result.

    The function loads the model at `model_path`, then for noise levels from 0.0
    to 1.0 (inclusive) with step `epsilon` it loads the MNIST test set corrupted by
    `noise_type` using `MnistDataloader.load_data`, evaluates the model
    on the noisy test set and records the model score (the metric returned by
    `model.evaluate`). The function displays a matplotlib figure (noise vs score)
    and returns the figure object along with the arrays of noise levels and scores.

    :param model_path: path to the saved Keras model
    :type model_path: str
    :param data_dir: directory where MNIST data is stored
    :type data_dir: str
    :param noise_type: type of noise to apply ('gaussian' or 'salt_and_pepper')
    :type noise_type: str
    :param epsilon: step for noise levels between 0.0 and 1.0 (inclusive)
    :type epsilon: float
    :param sample_size: optional integer to limit number of test samples evaluated (speeds up evaluation)
    :type sample_size: int or None
    :returns: (fig, noise_levels, scores) where `fig` is the matplotlib Figure,
              `noise_levels` is a list of noise factors and `scores` the corresponding model scores.
    :rtype: tuple
    """
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

    mnist_loader = MnistDataloader(data_dir=data_dir)

    noise_levels = np.arange(0.0, 1.0 + epsilon, epsilon)
    scores = []

    for level in noise_levels:
        print(f"Evaluating noise level {level:.3f}...")
        try:
            (_, _), (x_test_noisy, y_test) = mnist_loader.load_data(apply_noise=True, noise_type=noise_type, noise_factor=float(level))
        except Exception as e:
            print(f"Error loading noisy data for level {level}: {e}")
            scores.append(np.nan)
            continue

        # Convert to numpy array and reshape to (N,28,28,1)
        x_test_arr = np.array([np.array(img).reshape(28, 28, 1) for img in x_test_noisy])
        y_test_arr = np.array(y_test)

        # The noisy loader returns floats in [0,1]; if values exceed 1, normalize
        if x_test_arr.max() > 1.1:
            x_test_arr = x_test_arr.astype('float32') / 255.0
        else:
            x_test_arr = x_test_arr.astype('float32')

        # One-hot encode labels
        y_test_cat = to_categorical(y_test_arr, 10)

        # Optionally subsample to speed up evaluation
        if sample_size is not None and sample_size > 0 and sample_size < x_test_arr.shape[0]:
            idx = np.random.choice(x_test_arr.shape[0], sample_size, replace=False)
            x_eval = x_test_arr[idx]
            y_eval = y_test_cat[idx]
        else:
            x_eval = x_test_arr
            y_eval = y_test_cat

        # Evaluate silently
        result = model.evaluate(x_eval, y_eval, verbose=0)
        # result may be a scalar or list: choose metric if present
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            score = result[1]
        else:
            score = result
        scores.append(score)

    # Plot
    fig = plt.figure(figsize=(8, 6))
    plt.plot(noise_levels, scores, marker='o')
    plt.xlabel('Noise level')
    plt.ylabel('Model score')
    plt.title(f'Model performance vs noise (type={noise_type})')
    plt.grid(True)
    plt.show()

    return fig, noise_levels, np.array(scores)


if __name__ == '__main__':
    # evaluate_existing_model()
    evaluate_vs_noise(model_path='data/models/mnist_cnn.keras', data_dir='data/MNIST',
                      noise_type='salt_and_pepper', epsilon=0.05, sample_size=None)
