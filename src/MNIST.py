import numpy as np
import struct
from array import array
import os
from os.path  import join
import random
import matplotlib.pyplot as plt
import kagglehub
import shutil
from PIL import Image, ImageOps
import h5py

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
            data_dir = os.path.join(".", "data")
        self.data_dir = data_dir
        self.training_images_filepath = os.path.join(data_dir, 'MNIST',  'train-images-idx3-ubyte', 'train-images-idx3-ubyte')
        self.training_labels_filepath = os.path.join(data_dir, 'MNIST', 'train-labels-idx1-ubyte', 'train-labels-idx1-ubyte')
        self.test_images_filepath = os.path.join(data_dir, 'MNIST', 't10k-images-idx3-ubyte', 't10k-images-idx3-ubyte')
        self.test_labels_filepath = os.path.join(data_dir, 'MNIST', 't10k-labels-idx1-ubyte', 't10k-labels-idx1-ubyte')
    
    def download_mnist(self):
        """
        Downloads the MNIST dataset from Kaggle.

        :raises Exception: If the download or file operations fail.
        """
        print("Downloading MNIST dataset...")
        cache_path = kagglehub.dataset_download("hojjatk/mnist-dataset")
        path_dir = os.path.join(self.data_dir, "MNIST")
        # Create the target folder if it does not exist
        os.makedirs(path_dir, exist_ok=True)
        0
        # Copy the files to the target folder
        for item in os.listdir(cache_path):
            source = os.path.join(cache_path, item)
            destination = os.path.join(path_dir, item)
            
            if os.path.isfile(source):
                shutil.copy2(source, destination)
            elif os.path.isdir(source):
                shutil.copytree(source, destination, dirs_exist_ok=True)

    def read_images_labels(self, images_filepath, labels_filepath, num_images=None, random_selection=False):
        """Reads images and labels from specified files.

        :param images_filepath: path to the image file
        :param labels_filepath: path to the label file
        :param num_images: Number of images to load. If None, loads all images
        :param random_selection: If True, selects images randomly. If False, takes first num_images
        :returns: a tuple of images and labels
        :raises ValueError: if there is a magic number mismatch
        """
        if num_images is not None and num_images > 0:
            mode = "random" if random_selection else "sequential"
            print(f"Reading {num_images} images ({mode} selection)...")
        
        # Read labels file header to get size
        with open(labels_filepath, 'rb') as file:
            magic, total_size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            
            # Determine which indices to load
            if num_images is None:
                indices = list(range(total_size))
            elif random_selection:
                indices = sorted(np.random.choice(total_size, min(num_images, total_size), replace=False))
            else:
                indices = list(range(min(num_images, total_size)))
            
            # Read all labels first 
            all_labels = array("B", file.read())
            labels = [all_labels[i] for i in indices]
        
        # Read images file
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            
            images = []
            image_size = rows * cols
            
            if random_selection and num_images is not None and num_images > 0:
                # For random selection, we need to seek to specific positions
                for idx in indices:
                    file.seek(16 + idx * image_size)  # Skip header + offset to image
                    image_data = array("B", file.read(image_size))
                    img = np.array(image_data).reshape(rows, cols)
                    images.append(img)
            elif len(indices) > 0:
                # For sequential reading, read the required number of images
                num_to_read = len(indices)
                total_bytes = num_to_read * image_size
                image_data = array("B", file.read(total_bytes))
                
                for i in range(num_to_read):
                    img = np.array(image_data[i * image_size:(i + 1) * image_size])
                    img = img.reshape(rows, cols)
                    images.append(img)
        
        return images, labels
            
    def load_data(self, apply_noise=False, noise_type="gaussian", noise_factor=0.5, num_images_train=None, num_images_test=None, random_selection=False):
        """
        Loads training and test data, optionally with added noise.
        
        :param apply_noise: If True, applies noise to the data
        :param noise_type: Type of noise ('gaussian' or 'salt_and_pepper')
        :param noise_factor: Proportion/intensity of noise (0.0 to 1.0)
        :param num_images_train: Number of training images to load. If None, loads all images
        :param num_images_test: Number of test images to load. If None, loads all images
        :param random_selection: If True, selects images randomly. If False, takes first num_images
        :returns: Tuple of ((x_train, y_train), (x_test, y_test))
        """
        # Load only the required number of images directly from files
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, 
            self.training_labels_filepath,
            num_images=num_images_train,
            random_selection=random_selection
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, 
            self.test_labels_filepath,
            num_images=num_images_test,
            random_selection=random_selection
        )
        
        if not apply_noise:
            return (x_train, y_train), (x_test, y_test)
        
        # Normalize to float32
        x_train = np.array(x_train).astype('float32') / 255.0
        x_test = np.array(x_test).astype('float32') / 255.0
        
        x_train_noisy = x_train.copy()
        x_test_noisy = x_test.copy()

        if noise_type == "gaussian":
             # create a boolean mask for train where pixels will be changed (approx. noise_factor proportion)
            mask_train = np.random.random(x_train.shape) < noise_factor
            x_train_noisy[mask_train] += np.random.normal(loc=0.0, scale=1.0, size=np.sum(mask_train))

    
            # create a boolean mask for test where pixels will be changed (approx. noise_factor proportion)
            mask_test = np.random.random(x_test.shape) < noise_factor
            x_test_noisy[mask_test] += np.random.normal(loc=0.0, scale=1.0, size=np.sum(mask_test))

            x_train_noisy = np.clip(x_train_noisy, 0., 1.)
            x_test_noisy = np.clip(x_test_noisy, 0., 1.)
        
        elif noise_type == "salt_and_pepper":
            # create a mask for salt and pepper noise
            mask_train = np.random.random(x_train.shape) < noise_factor
            mask_test = np.random.random(x_test.shape) < noise_factor

            # Salt (1.0) and Pepper (0.0) noise
            x_train_noisy[mask_train] = np.random.choice([0.0, 1.0], size=np.sum(mask_train))
            x_test_noisy[mask_test] = np.random.choice([0.0, 1.0], size=np.sum(mask_test))

        return (x_train_noisy, y_train), (x_test_noisy, y_test)

    def create_sequence(self, mnist_images, mnist_labels, num_digits=4):
        """
        Creates a CAPTCHA-like sequence of MNIST digits with random transformations.
        
        :param mnist_images: Array of MNIST images
        :param mnist_labels: Array of MNIST labels
        :param num_digits: Number of digits to concatenate (default: 4)
        :returns: Tuple of (final_image, label_arrays) where final_image is a PIL Image
                  and label_arrays is a list of the digit labels
        """
        canvas_height = 100
        canvas_width = 300 
        
        final_height = canvas_height
        final_width = 110

        working_canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
        
        current_x = 10 
        label_arrays = []
        
        for _ in range(num_digits):
            choice = random.randint(0, len(mnist_images) - 1)
            img_array = mnist_images[choice]
            label_arrays.append(mnist_labels[choice])

            if img_array.max() <= 1.0: 
                img_array = (img_array * 255).astype(np.uint8)
            else: 
                img_array = img_array.astype(np.uint8)
            digit_img = Image.fromarray(img_array, mode="L")
            
            # Homothetic transformation and rotation
            scale = random.uniform(0.9, 1.3)
            digit_img = digit_img.resize(
                (int(digit_img.width * scale), int(digit_img.height * scale)), 
                resample=Image.BILINEAR
            )
            angle = random.uniform(-20, 20)
            digit_img = digit_img.rotate(angle, resample=Image.BILINEAR, expand=True)

            # We paste the digit onto a transparent canvas 
            digit_rgba = Image.new("RGBA", digit_img.size, (255, 255, 255, 0))
            digit_rgba.paste((255, 255, 255, 255), (0, 0), mask=digit_img)

            bbox = digit_rgba.getbbox()  # Get bounding box of non-transparent pixels
            if bbox:
                digit_width = bbox[2] - bbox[0]
                digit_height = bbox[3] - bbox[1]
                y_pos = (canvas_height - digit_height) // 2  # Center vertically
                x_offset_correction = bbox[0]  # Leftmost non-transparent pixel
                
                paste_x = current_x - x_offset_correction  # Correct paste position
                working_canvas.alpha_composite(digit_rgba, dest=(paste_x, y_pos))
                
                overlap = random.randint(-6, -2) 
                current_x += digit_width - overlap  # Update current_x for next digit

        visible_bbox = working_canvas.getbbox()
        
        if visible_bbox:
            padding = 2
            left, top, right, bottom = visible_bbox
            crop_box = (
                max(0, left - padding),
                max(0, top - padding),
                min(canvas_width, right + padding),
                min(canvas_height, bottom + padding)
            )
            tight_image = working_canvas.crop(crop_box)
            # Scale the image to fit into final dimensions
            tight_image = tight_image.resize((final_width, final_height), resample=Image.BILINEAR)
            final_image = Image.new("RGB", tight_image.size, (0, 0, 0))
            final_image.paste(tight_image, (0, 0), mask=tight_image)
            return final_image, label_arrays
        else:
            return Image.new("RGB", (10, 10), (0, 0, 0)), [-1, -1, -1, -1]
    
    def create_captcha_dataset(self, num_train=10000, num_test=2000, num_digits=4, output_dir='captcha_data', seed=None):
        """
        Creates a dataset of CAPTCHA-like images composed of MNIST digits and saves to H5 format.

        :param num_train: Number of training CAPTCHA images to generate
        :param num_test: Number of test CAPTCHA images to generate
        :param num_digits: Number of digits in each CAPTCHA (default: 4)
        :param output_dir: Directory to save the generated H5 files
        :param seed: Random seed for reproducibility (optional)
        :returns: Paths to the generated H5 files
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        (X_train, y_train), (X_test, y_test) = self.load_data()
        X_train = np.array([np.array(img).reshape(28, 28) for img in X_train])
        X_test = np.array([np.array(img).reshape(28, 28) for img in X_test])
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        output_dir = os.path.join(self.data_dir, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        captcha_file = os.path.join(output_dir, 'captcha_dataset.h5')
        
        print(f"\n GGenerating {num_train} training CAPTCHAs...")
        train_images = []
        train_labels = []
        
        for i in range(num_train):
            image, labels = self.create_sequence(X_train, y_train, num_digits=num_digits)
            # Convertir PIL Image en numpy array
            img_array = np.array(image)
            train_images.append(img_array)
            train_labels.append(labels)
            
            if (i + 1) % 1000 == 0:
                print(f"  Progress: {i + 1}/{num_train}")
        
        print(f"\n GGenerating {num_test} test CAPTCHAs...")
        test_images = []
        test_labels = []
        
        for i in range(num_test):
            image, labels = self.create_sequence(X_test, y_test, num_digits=num_digits)
            # Convertir PIL Image en numpy array
            img_array = np.array(image)
            test_images.append(img_array)
            test_labels.append(labels)
            
            if (i + 1) % 500 == 0:
                print(f"  Progress: {i + 1}/{num_test}")
        
        # Convert to numpy arrays
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)
        
        # Save in H5 format
        print(f"Saving to {captcha_file}...")
        with h5py.File(captcha_file, 'w') as f:
            # Training data
            f.create_dataset('X_train', data=train_images, compression='gzip')
            f.create_dataset('y_train', data=train_labels, compression='gzip')
            
            # Test data
            f.create_dataset('X_test', data=test_images, compression='gzip')
            f.create_dataset('y_test', data=test_labels, compression='gzip')
            
            # Metadata
            f.attrs['num_digits'] = num_digits
            f.attrs['num_train'] = num_train
            f.attrs['num_test'] = num_test
            f.attrs['image_shape'] = train_images.shape[1:]
        
        print(f" CAPTCHA dataset created successfully!")
        
        return captcha_file
    
    def load_captcha_dataset(self, h5_filepath=None, num_images_train=None, num_images_test=None, random_selection=False):
        """
        Loads the CAPTCHA dataset from an H5 file.

        :param h5_filepath: Path to the H5 file containing the CAPTCHA dataset
        :param num_images_train: Number of training images to load. If None, loads all training images
        :param num_images_test: Number of test images to load. If None, loads all test images
        :param random_selection: If True, selects images randomly. If False, takes first num_images
        :returns: Tuple of ((X_train, y_train), (X_test, y_test))
        """
        if h5_filepath is None:
            h5_filepath = os.path.join(self.data_dir, 'captcha_data', 'captcha_dataset.h5')
        
        with h5py.File(h5_filepath, 'r') as f:
            # Get the total size without loading data
            total_train = f['X_train'].shape[0]
            total_test = f['X_test'].shape[0]
            
            # Handle training data
            if num_images_train is None:
                # Load all training data
                print(f"Loading all training data ({total_train} images)...")
                X_train = np.array(f['X_train'])
                y_train = np.array(f['y_train'])
            else:
                if random_selection:
                    # Random selection - sorted to optimize H5 reading
                    num_to_load = min(num_images_train, total_train)
                    print(f"Loading {num_to_load} training images (random selection)...")
                    train_indices = sorted(np.random.choice(total_train, num_to_load, replace=False))
                    # Load X and y with same indices to ensure correspondence
                    X_train = f['X_train'][train_indices]
                    y_train = f['y_train'][train_indices]
                else:
                    # Sequential selection - slice directly (very efficient with H5)
                    num_train = min(num_images_train, total_train)
                    print(f"Loading {num_train} training images (sequential)...")
                    X_train = f['X_train'][:num_train]
                    y_train = f['y_train'][:num_train]
            
            # Handle test data
            if num_images_test is None:
                # Load all test data
                print(f"Loading all test data ({total_test} images)...")
                X_test = np.array(f['X_test'])
                y_test = np.array(f['y_test'])
            else:
                if random_selection:
                    # Random selection - sorted to optimize H5 reading
                    num_to_load = min(num_images_test, total_test)
                    print(f"Loading {num_to_load} test images (random selection)...")
                    test_indices = sorted(np.random.choice(total_test, num_to_load, replace=False))
                    # Load X and y with same indices to ensure correspondence
                    X_test = f['X_test'][test_indices]
                    y_test = f['y_test'][test_indices]
                else:
                    # Sequential selection - slice directly (very efficient with H5)
                    num_test = min(num_images_test, total_test)
                    print(f"Loading {num_test} test images (sequential)...")
                    X_test = f['X_test'][:num_test]
                    y_test = f['y_test'][:num_test]
        
        print(f"CAPTCHA dataset loaded successfully!")
        return (X_train, y_train), (X_test, y_test)
    
    def show_captcha(self, num_train=5, num_test=3, h5_filepath=None, seed=None):
        """
        Displays CAPTCHA images from the dataset with their labels.

        :param num_train: Number of training CAPTCHA images to display
        :param num_test: Number of test CAPTCHA images to display
        :param h5_filepath: Optional path to the H5 file containing the CAPTCHA dataset
        :param seed: Random seed for reproducibility (optional)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Load only the requested number of images with random selection
        (X_train, y_train), (X_test, y_test) = self.load_captcha_dataset(
            h5_filepath=h5_filepath,
            num_images_train=num_train if num_train > 0 else 0,
            num_images_test=num_test if num_test > 0 else 0,
            random_selection=True
        )
        
        images_to_show = []
        titles_to_show = []
        
        # Display all loaded training images
        if num_train > 0:
            for idx in range(len(X_train)):
                images_to_show.append(X_train[idx])
                label_str = ''.join(map(str, y_train[idx]))
                titles_to_show.append(f'Train = {label_str}')
        
        # Display all loaded test images
        if num_test > 0:
            for idx in range(len(X_test)):
                images_to_show.append(X_test[idx])
                label_str = ''.join(map(str, y_test[idx]))
                titles_to_show.append(f'Test = {label_str}')
        
        # Display
        total_images = len(images_to_show)
        cols = 5
        rows = (total_images + cols - 1) // cols  
        
        plt.figure(figsize=(20, 4 * rows))
        for index, (image, title) in enumerate(zip(images_to_show, titles_to_show), 1):
            plt.subplot(rows, cols, index)
            plt.imshow(image, cmap=plt.cm.gray)
            plt.title(title, fontsize=12)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Displaying {num_train} training CAPTCHAs and {num_test} test CAPTCHAs")
    
    def show_images(self, num_train=10, num_test=5, show_noisy=False, noise_type="gaussian", noise_factor=0.5, seed=None):
        """
        Displays a collection of MNIST images with titles. 
        Can display original images or images with added noise.

        :param num_train: Number of training images to display
        :param num_test: Number of test images to display
        :param seed: Random seed for reproducibility (optional)
        :param show_noisy: If True, loads data with noise using the specified parameters.
        :param noise_type: Type of noise ('gaussian' or 'salt_and_pepper').
        :param noise_factor: Intensity of the noise (0.0 to 1.0).
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Load only the requested number of images with random selection
        if show_noisy:
            print(f"Loading noisy data with ({noise_type} noise, factor: {noise_factor})...")
            try:
                (x_train, y_train), (x_test, y_test) = self.load_data(
                    apply_noise=True, 
                    noise_type=noise_type, 
                    noise_factor=noise_factor,
                    num_images_train=num_train if num_train > 0 else 0,
                    num_images_test=num_test if num_test > 0 else 0,
                    random_selection=True
                )
                title_suffix = f"\n({noise_type})"
            except Exception as e:
                print(f"Error loading noisy data: {e}, downloading MNIST dataset")
                self.download_mnist()
                (x_train, y_train), (x_test, y_test) = self.load_data(
                    apply_noise=True, 
                    noise_type=noise_type, 
                    noise_factor=noise_factor,
                    num_images_train=num_train if num_train > 0 else 0,
                    num_images_test=num_test if num_test > 0 else 0,
                    random_selection=True
                )
                title_suffix = f"\n({noise_type})"
        else:
            print("Loading original data...")
            try:
                (x_train, y_train), (x_test, y_test) = self.load_data(
                    num_images_train=num_train if num_train > 0 else 0,
                    num_images_test=num_test if num_test > 0 else 0,
                    random_selection=True
                )
                title_suffix = ""
            except Exception as e:
                print(f"Error loading data: {e}, downloading MNIST dataset")
                self.download_mnist()
                (x_train, y_train), (x_test, y_test) = self.load_data(
                    num_images_train=num_train if num_train > 0 else 0,
                    num_images_test=num_test if num_test > 0 else 0,
                    random_selection=True
                )
                title_suffix = ""

        images_2_show = []
        titles_2_show = []
        
        # Display all loaded training images
        for i in range(len(x_train)):
            images_2_show.append(x_train[i])
            titles_2_show.append('training image = ' + str(y_train[i]) + title_suffix)    

        # Display all loaded test images
        for i in range(len(x_test)):
            images_2_show.append(x_test[i])        
            titles_2_show.append('test image = ' + str(y_test[i]) + title_suffix)  

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
        print(f"Displaying {num_train} training images and {num_test} test images")
        plt.show()
        
        


if __name__ == '__main__':
    loader = MnistDataloader()
    
    # Example 1: Display normal MNIST images
    loader.show_images(num_train=9, num_test=0,seed=42)
    
    # Example 2: Display MNIST images with noise
    # loader.show_images(num_train=10, num_test=5, show_noisy=True, noise_type="salt_and_pepper", noise_factor=0.3, seed=42)
    
    # Exemple 3: Créer un dataset CAPTCHA
    # loader.create_captcha_dataset(num_train=100_000, num_test=20_000, seed=42)
    
    # Exemple 4: Afficher des CAPTCHAs
    # loader.show_captcha(num_train=5, num_test=3, seed=42)
    
    pass