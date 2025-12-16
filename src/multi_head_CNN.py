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

def apply_balanced_noise(images, noise_type='salt_and_pepper', noise_factor=0.3, noise_factor_end=None, step=0.1, rgb_noise=False):
    """Applies noise with balanced distribution across different noise levels.
    
    :param images: Array of images to add noise to
    :param noise_type: Type of noise ('gaussian' or 'salt_and_pepper')
    :param noise_factor: Starting noise factor
    :param noise_factor_end: Ending noise factor (if None, all images use noise_factor)
    :param step: Step size between noise levels
    :param rgb_noise: If True, applies noise independently per RGB channel
    :returns: Array of noisy images
    """
    from MNIST import MnistDataloader
    temp_loader = MnistDataloader()
    
    if noise_factor_end is None:
        # Apply same noise to all images
        print(f"Applying {noise_type} noise (factor: {noise_factor})...")
        return temp_loader.add_noise(images, noise_type, noise_factor, rgb_noise)
    
    # Calculate noise levels
    noise_levels = []
    current = noise_factor
    while current <= noise_factor_end + 1e-9: 
        noise_levels.append(round(current, 5))
        current += step
    
    if len(noise_levels) == 0:
        raise ValueError("No valid noise levels generated. Check your parameters.")
    
    print(f"Applying {noise_type} noise with {len(noise_levels)} levels: {noise_levels}")
    
    # Distribute images evenly across noise levels
    num_images = len(images)
    images_per_level = num_images // len(noise_levels)
    remainder = num_images % len(noise_levels)
    
    noisy_images = []
    current_idx = 0
    
    for i, level in enumerate(noise_levels):
        # Calculate how many images for this level (distribute remainder)
        num_for_level = images_per_level + (1 if i < remainder else 0)
        
        if num_for_level > 0:
            # Extract images for this noise level
            batch = images[current_idx:current_idx + num_for_level]
            
            # Apply noise
            print(f"  Level {level:.2f}: processing {num_for_level} images ({current_idx+1}-{current_idx+num_for_level})")
            batch_noisy = temp_loader.add_noise(images=batch, noise_type=noise_type, noise_factor=level, rgb_noise=rgb_noise)
            noisy_images.append(batch_noisy)
            
            current_idx += num_for_level
    
    # Concatenate all batches
    return np.concatenate(noisy_images, axis=0)

def train_and_evaluate(data_dir='data', model_save_path='data/models/multi_output_cnn.keras',
                      apply_noise=False, noise_type='gaussian', noise_factor=0.3, 
                      noise_factor_end=None, noise_step=0.1, rgb_noise=False):
    """Trains and evaluates the CNN model.

    :param data_dir: The directory where the MNIST data is stored.
    :param model_save_path: The path to save the trained model.
    :param apply_noise: If True, applies noise to the loaded CAPTCHA dataset
    :param noise_type: Type of noise ('gaussian' or 'salt_and_pepper')
    :param noise_factor: Starting noise factor (0.0 to 1.0)
    :param noise_factor_end: Ending noise factor (if None, uses only noise_factor)
    :param noise_step: Step size between noise levels
    :param rgb_noise: If True, applies noise independently per RGB channel
    :raises Exception: if data loading or model training fails
    """

    mnist_loader = MnistDataloader(data_dir=data_dir)
    try:
        # Load dataset without noise first
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
    
    # Apply noise if requested (before converting to grayscale)
    if apply_noise:
        noise_mode = "RGB" if rgb_noise else "grayscale"
        if noise_factor_end is None:
            print(f"\nApplying {noise_type} noise ({noise_mode} mode) with factor {noise_factor}")
        else:
            print(f"\nApplying {noise_type} noise ({noise_mode} mode) from {noise_factor} to {noise_factor_end} (step {noise_step})")
        
        x_train = apply_balanced_noise(x_train, noise_type, noise_factor, noise_factor_end, noise_step, rgb_noise)
        x_test = apply_balanced_noise(x_test, noise_type, noise_factor, noise_factor_end, noise_step, rgb_noise)
    
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
    # Example 1: Train without noise (original)
    # train_and_evaluate()
    
    # Example 2: Train with uniform noise (same factor for all images)
    # train_and_evaluate(apply_noise=True, noise_type='gaussian', noise_factor=0.3)
    
    # Example 3: Train with balanced noise range (1/3 at 0.3, 1/3 at 0.4, 1/3 at 0.5)
    # train_and_evaluate(
    #     apply_noise=True, 
    #     noise_type='gaussian', 
    #     noise_factor=0.3, 
    #     noise_factor_end=0.5, 
    #     noise_step=0.1,
    #     rgb_noise=False
    # )
    
    # Example 4: Train with salt_and_pepper noise with range
    train_and_evaluate(
        model_save_path='data/models/multi_output_cnn_sandp.keras',
        apply_noise=True, 
        noise_type='salt_and_pepper', 
        noise_factor=0.2, 
        noise_factor_end=0.4, 
        noise_step=0.1,
        rgb_noise=False
    )
