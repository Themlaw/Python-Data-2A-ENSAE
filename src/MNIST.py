

import numpy as np
import struct
from array import array
import os
from os.path  import join
import random
import matplotlib.pyplot as plt
import kagglehub
import shutil

class MnistDataloader(object):
    """
    This class handles the MNIST dataset.
    """
    def __init__(self, data_dir=None,):
        """
        Initializes the dataloader, setting up file paths for the training and test data.

        :param data_dir: Optional string specifying the directory of the MNIST data.
        :type data_dir: str or None
        """
        if data_dir is None:
            data_dir = os.path.join(".", "data", "MNIST")
        self.data_dir = data_dir
        self.training_images_filepath = os.path.join(data_dir, 'train-images-idx3-ubyte', 'train-images-idx3-ubyte')
        self.training_labels_filepath = os.path.join(data_dir, 'train-labels-idx1-ubyte', 'train-labels-idx1-ubyte')
        self.test_images_filepath = os.path.join(data_dir, 't10k-images-idx3-ubyte', 't10k-images-idx3-ubyte')
        self.test_labels_filepath = os.path.join(data_dir, 't10k-labels-idx1-ubyte', 't10k-labels-idx1-ubyte')
    
    def download_mnist(self):
        """
        Downloads the MNIST dataset from Kaggle.

        :raises Exception: If the download or file operations fail.
        """
        cache_path = kagglehub.dataset_download("hojjatk/mnist-dataset")
        
        # Create the target folder if it does not exist
        os.makedirs(self.data_dir, exist_ok=True)
        0
        # Copy the files to the target folder
        for item in os.listdir(cache_path):
            source = os.path.join(cache_path, item)
            destination = os.path.join(self.data_dir, item)
            
            if os.path.isfile(source):
                shutil.copy2(source, destination)
            elif os.path.isdir(source):
                shutil.copytree(source, destination, dirs_exist_ok=True)

    def read_images_labels(self, images_filepath, labels_filepath):
        """Reads images and labels from specified files.

        :param images_filepath: path to the image file
        :param labels_filepath: path to the label file
        :returns: a tuple of images and labels
        :raises ValueError: if there is a magic number mismatch
        """
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        """
        Loads the training and test data

        :returns: A tuple containing two tuples: (x_train, y_train) and (x_test, y_test).
                  Each inner tuple contains the images and labels for the respective set.
        :rtype: tuple
        :raises Exception: If the data files are not found or cannot be read.
        """
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)    
    
    def load_data_with_noise(self, noise_type="gaussian", noise_factor=0.5):
        """
        Loads the training and test data with added noise.

        :param noise_type: Type of noise to apply. Supported values: 'gaussian' (Gaussian noise applied to a fraction of pixels)
                           or 'salt_and_pepper' (random pixels set to 0.0 or 1.0).
        :type noise_type: str
        :param noise_factor: Proportion of pixels to affect by noise, between 0.0 and 1.0. For 'gaussian', defines the probability
                            that a pixel receives noise. For 'salt_and_pepper', defines the proportion of pixels replaced.
        :type noise_factor: float
        :returns: A tuple containing two tuples: (x_train_noisy, y_train) and (x_test_noisy, y_test).
                  Each inner tuple contains the noisy images (normalized to [0., 1.] as float32) and original labels for the respective set.
        :rtype: tuple
        :raises Exception: If the data files are not found or cannot be read.
        """
        if noise_type == "gaussian":
            (x_train, y_train), (x_test, y_test) = self.load_data()

            x_train = np.array(x_train).astype('float32') / 255.0
            x_test = np.array(x_test).astype('float32') / 255.0

            x_train_noisy = x_train.copy()
            x_test_noisy = x_test.copy()
    
            # create a boolean mask for train where pixels will be changed (approx. noise_factor proportion)
            mask_train = np.random.random(x_train.shape) < noise_factor
            x_train_noisy[mask_train] += np.random.normal(loc=0.0, scale=1.0, size=np.sum(mask_train))
    
            # create a boolean mask for test where pixels will be changed (approx. noise_factor proportion)
            mask_test = np.random.random(x_test.shape) < noise_factor
            x_test_noisy[mask_test] += np.random.normal(loc=0.0, scale=1.0, size=np.sum(mask_test))

            x_train_noisy = np.clip(x_train_noisy, 0., 1.)
            x_test_noisy = np.clip(x_test_noisy, 0., 1.)

            return (x_train_noisy, y_train), (x_test_noisy, y_test)
        
        elif noise_type == "salt_and_pepper":
            (x_train, y_train), (x_test, y_test) = self.load_data()

            x_train = np.array(x_train).astype('float32') / 255.0
            x_test = np.array(x_test).astype('float32') / 255.0

            x_train_noisy = x_train.copy()
            x_test_noisy = x_test.copy()

            # create boolean masks where pixels will be changed (approx. noise_factor proportion)
            mask_train = np.random.random(x_train.shape) < noise_factor
            mask_test = np.random.random(x_test.shape) < noise_factor

            # assign salt (1.0) or pepper (0.0) to the selected pixels
            num_train = np.sum(mask_train)
            if num_train > 0:
                vals = np.random.choice([0.0, 1.0], size=num_train)
                x_train_noisy[mask_train] = vals

            num_test = np.sum(mask_test)
            if num_test > 0:
                vals = np.random.choice([0.0, 1.0], size=num_test)
                x_test_noisy[mask_test] = vals

            return (x_train_noisy, y_train), (x_test_noisy, y_test)


    


input_path = 'data\\MNIST'

def show_images(show_noisy=False, noise_type="gaussian", noise_factor=0.5):
    """
    Displays a collection of images with titles. 
    Can display original images or images with added noise.

    :param show_noisy: If True, loads data with noise using the specified parameters.
    :param noise_type: Type of noise ('gaussian' or 'salt_and_pepper').
    :param noise_factor: Intensity of the noise (0.0 to 1.0).
    """
    mnist_dataloader = MnistDataloader(data_dir=input_path)
    
    # Choice of data loading according to the selected option
    if show_noisy:
        print(f"Chargement des données avec bruit ({noise_type}, facteur: {noise_factor})...")
        (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data_with_noise(noise_type, noise_factor)
        title_suffix = f"\n({noise_type})"
    else:
        print("Chargement des données originales...")
        (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
        title_suffix = ""

    images_2_show = []
    titles_2_show = []
    
    # Random selection from the training set
    # Using len() to avoid out-of-bounds index errors
    for i in range(0, 10):
        r = random.randint(0, len(x_train) - 1)
        images_2_show.append(x_train[r])
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]) + title_suffix)    

    # Random selection from the test set
    for i in range(0, 5):
        r = random.randint(0, len(x_test) - 1)
        images_2_show.append(x_test[r])        
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]) + title_suffix)  

    cols = 5
    rows = int(len(images_2_show)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images_2_show, titles_2_show):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15)        
        index += 1
    plt.show()


if __name__ == '__main__':
    show_images(show_noisy=True, noise_type="gaussian", noise_factor=0.3)