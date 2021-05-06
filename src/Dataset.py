import numpy as np
import tensorflow as tf
from tensorflow import newaxis, clip_by_value
from tensorflow.keras.datasets import fashion_mnist


class Dataset:
    def __init__(self, noise_factor=0.2):
        try:
            self.train = np.load('dataset/mnist_fashion_32_train.npy')
            self.test = np.load('dataset/mnist_fashion_32_test.npy')
        except FileNotFoundError:
            (train_original, _), (test_original, _) = fashion_mnist.load_data()
            train_original = train_original.astype('float32') / 255.
            test_original = test_original.astype('float32') / 255.

            self.train = np.zeros((len(train_original), 32, 32))
            self.test = np.zeros((len(test_original), 32, 32))

            # Copy values.
            for image_index in range(len(train_original)):
                for row_index in range(len(train_original[0])):
                    for column_index in range(len(train_original[0][0])):
                        self.train[image_index][row_index + 2][column_index + 2] = \
                            train_original[image_index][row_index][column_index]

            # Copy values.
            for image_index in range(len(test_original)):
                for row_index in range(len(test_original[0])):
                    for column_index in range(len(test_original[0][0])):
                        self.test[image_index][row_index + 2][column_index + 2] = \
                            test_original[image_index][row_index][column_index]

            self.train = self.train[..., newaxis]
            self.test = self.test[..., newaxis]

            np.save('dataset/mnist_fashion_32_train.npy', self.train)
            np.save('dataset/mnist_fashion_32_test.npy', self.test)

        self.train_noisy = self.train + noise_factor * tf.random.normal(shape=self.train.shape)
        self.test_noisy = self.test + noise_factor * tf.random.normal(shape=self.test.shape)
        self.train_noisy = clip_by_value(self.train_noisy, clip_value_min=0., clip_value_max=1.)
        self.test_noisy = clip_by_value(self.test_noisy, clip_value_min=0., clip_value_max=1.)
