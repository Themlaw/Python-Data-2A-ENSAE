

import numpy as np # linear algebra
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
        
        # Créer le dossier cible s'il n'existe pas
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Copier les fichiers vers le dossier cible
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

    


input_path = 'data\\MNIST'

def show_images():
    """Displays a collection of images with titles.

    :param images: a list of images to display
    :param title_texts: a list of titles for the images
    :raises Exception: if data loading or plotting fails
    """
    mnist_dataloader = MnistDataloader(data_dir=input_path)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(1, 60000)
        images_2_show.append(x_train[r])
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

    for i in range(0, 5):
        r = random.randint(1, 10000)
        images_2_show.append(x_test[r])        
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))  

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
            plt.title(title_text, fontsize = 15);        
        index += 1
    plt.show()


if __name__ == '__main__':
    show_images()