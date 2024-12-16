import os
import requests
import io
import numpy as np

class MNISTLoader:
    def __init__(self):
        self.data_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.mnist_path = os.path.join(self.data_dir_path, "MNIST")

        self._normalize_image = False
        self._specific_digit = None

        # Build a data folder if needed
        if not os.path.exists(self.data_dir_path):
            os.mkdir(self.data_dir_path)

    def set_normalize_image(self):
        self._normalize_image = True

    def set_specific_digit(self, digit):
        if not (0 <= digit <= 9):
            raise ValueError("specific_digit must be between 0 and 9.")

        self._specific_digit = digit

    #########################################################################
    #                       LOADING FUNCTIONS                               #
    #########################################################################
    def load_mnist_data(self, specific_digit=None):

        # Download if needed
        if not os.path.exists(self.mnist_path):
            self._download_mnist_data()

        x_train = np.load(os.path.join(self.mnist_path, "x_train.npy"))
        x_test = np.load(os.path.join(self.mnist_path, "x_test.npy"))
        y_train = np.load(os.path.join(self.mnist_path, "y_train.npy"))
        y_test = np.load(os.path.join(self.mnist_path, "y_test.npy"))

        # Apply enabled pre-process
        if specific_digit is not None:
            x_train, y_train, x_test, y_test = self._use_specific_digit(x_train, y_train, x_test, y_test)
        if self._normalize_image:
            x_train, x_test = self._normalize_images(x_train, x_test)

        return x_train, y_train, x_test, y_test

    def _download_mnist_data(self ):
        print("Downloading MNIST dataset...")
        response = requests.get('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz')
        response.raise_for_status()
        data = np.load(io.BytesIO(response.content))

        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']

        # Build a MNIST folder if needed
        if not os.path.exists(self.mnist_path):
            os.mkdir(self.mnist_path)

        np.save(os.path.join(self.mnist_path, "x_train.npy"), x_train)
        np.save(os.path.join(self.mnist_path, "y_train.npy"), y_train)
        np.save(os.path.join(self.mnist_path, "x_test.npy"), x_test)
        np.save(os.path.join(self.mnist_path, "y_test.npy"), y_test)

    #########################################################################
    #                  Pre-Processing FUNCTIONS                             #
    #########################################################################
    def _normalize_images(self, x_train, x_test):
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        return x_train, x_test

    def _use_specific_digit(self, x_train, y_train, x_test, y_test):
        # If a specific digit is provided, filter the data to only include samples of that digit
        if self._specific_digit is not None:
            # Filter the training and test data
            train_mask = (y_train == self._specific_digit)
            test_mask = (y_test == self._specific_digit)

            x_train = x_train[train_mask]
            y_train = y_train[train_mask]
            x_test = x_test[test_mask]
            y_test = y_test[test_mask]

        return x_train, y_train, x_test, y_test


