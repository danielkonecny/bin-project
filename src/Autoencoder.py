from abc import ABC
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class Autoencoder(Model, ABC):
    def __init__(self, encoder=None, decoder=None):
        super(Autoencoder, self).__init__()
        if encoder is None and decoder is None:
            self.encoder = tf.keras.Sequential([
                layers.Input(shape=(32, 32, 1)),
                layers.Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
                layers.Conv2D(8, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')
            ])

            self.decoder = tf.keras.Sequential([
                layers.Conv2DTranspose(8, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
                layers.Conv2DTranspose(16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
                layers.Conv2D(1, kernel_size=(3, 3), padding='same', activation='sigmoid')
            ])
        elif encoder is not None and decoder is not None:
            self.encoder = encoder
            self.decoder = decoder
        else:
            print("ERROR: Bad argument combination for initializing autoencoder.")

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
