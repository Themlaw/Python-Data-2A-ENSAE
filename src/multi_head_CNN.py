import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, MaxPooling2D
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
    #We separate the 4 labels for multi-output
    y_train1 = to_categorical([y_train[i][0] for i in range(len(y_train))],10)
    y_train2 = to_categorical([y_train[i][1] for i in range(len(y_train))],10)
    y_train3 = to_categorical([y_train[i][2] for i in range(len(y_train))],10)
    y_train4 = to_categorical([y_train[i][3] for i in range(len(y_train))],10)
    y_train_list = np.array([y_train1, y_train2, y_train3, y_train4])
    
    y_test1 = to_categorical([y_test[i][0] for i in range(len(y_test))],10)
    y_test2 = to_categorical([y_test[i][1] for i in range(len(y_test))],10)
    y_test3 = to_categorical([y_test[i][2] for i in range(len(y_test))],10)
    y_test4 = to_categorical([y_test[i][3] for i in range(len(y_test))],10)
    y_test_list = np.array([y_test1, y_test2, y_test3, y_test4])

    return (x_train, y_train_list), (x_test, y_test_list)

def build_model():
    """Builds and compiles a multi_output CNN model.

    :returns: a compiled Keras model
    """
    # Input layer
    inputs = Input(shape=(100, 110, 1))
    
    # Shared convolutional base
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x) 
    x = MaxPooling2D((2, 2))(x)  
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    
    # Four separate output heads for each digit
    digit1 = Dense(64, activation='relu', name='digit1_dense')(x)
    digit1 = Dense(10, activation='softmax', name='digit1')(digit1)
    
    digit2 = Dense(64, activation='relu', name='digit2_dense')(x)
    digit2 = Dense(10, activation='softmax', name='digit2')(digit2)
    
    digit3 = Dense(64, activation='relu', name='digit3_dense')(x)
    digit3 = Dense(10, activation='softmax', name='digit3')(digit3)
    
    digit4 = Dense(64, activation='relu', name='digit4_dense')(x)
    digit4 = Dense(10, activation='softmax', name='digit4')(digit4)
    
    # Create model with multiple outputs
    model = Model(inputs=inputs, outputs=[digit1, digit2, digit3, digit4])
    
    model.compile(
        optimizer='adam',
        loss=['categorical_crossentropy', 'categorical_crossentropy', 
              'categorical_crossentropy', 'categorical_crossentropy'],
        metrics=['accuracy','accuracy','accuracy','accuracy']
    )
    return model

def train_and_evaluate(data_dir='data', model_save_path='data/models/multi_output_cnn.keras'):
    """Trains and evaluates the CNN model.

    :param data_dir: The directory where the MNIST data is stored.
    :param model_save_path: The path to save the trained model.
    :raises Exception: if data loading or model training fails
    """

    mnist_loader = MnistDataloader(data_dir=data_dir)
    try:
        (x_train, y_train), (x_test, y_test) = mnist_loader.load_captcha_dataset()
    except Exception as e:
        print(f"Captcha dataset not found with error: {e}, creating a new one...")
        try:
            mnist_loader.create_captcha_dataset(num_train=100_000, num_test=10_000)
            (x_train, y_train), (x_test, y_test) = mnist_loader.load_captcha_dataset()
        except Exception as e2:
            print(f"Failed to create captcha dataset with error: {e2}, downloading MNIST data...")
            mnist_loader.download_mnist()
            mnist_loader.create_captcha_dataset(num_train=100_000, num_test=10_000)
            (x_train, y_train), (x_test, y_test) = mnist_loader.load_captcha_dataset()
    
    # Convert RGB to grayscale and reshape to (100, 110, 1)
    x_train = np.array([np.mean(img, axis=-1, keepdims=True) for img in x_train])
    x_test = np.array([np.mean(img, axis=-1, keepdims=True) for img in x_test])
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    (x_train, y_train), (x_test, y_test) = preprocess_data(x_train, y_train, x_test, y_test)

    model = build_model()
    model.summary()

    print("\nTraining model...")
    model.fit(x_train, [y_train[0], y_train[1], y_train[2], y_train[3]], 
              epochs=5, batch_size=64, 
              validation_data=(x_test, [y_test[0], y_test[1], y_test[2], y_test[3]]))

    print("\nEvaluating model...")
    results = model.evaluate(x_test, [y_test[0], y_test[1], y_test[2], y_test[3]])
    print(f'\nTest results: {results}')
    print(f'Digit 1 accuracy: {results[5]:.4f}')
    print(f'Digit 2 accuracy: {results[6]:.4f}')
    print(f'Digit 3 accuracy: {results[7]:.4f}')
    print(f'Digit 4 accuracy: {results[8]:.4f}')

    # Create the data/models folder if it does not exist.
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to : {model_save_path}")

if __name__ == '__main__':
    train_and_evaluate()
