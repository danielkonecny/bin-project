"""Autoencoder Design
Design Autoencoder for image denoising using Evolutionary Algorithm (Genetic Algorithm).
Course: Bio-Inspired Computers (BIN)
Organisation: Brno University of Technology - Faculty of Information Technologies
Author: Daniel Konecny (xkonec75)
File: Transformer.py
Date: 07. 05. 2021
"""


import tensorflow as tf


def chromosome_to_encoder(chromosome):
    encoder = tf.keras.Sequential()
    encoder.add(tf.keras.layers.Input(shape=(32, 32, 1)))

    for gene in chromosome:
        encoder.add(gene_to_conv2d(gene))

    return encoder


def chromosome_to_decoder(chromosome):
    decoder = tf.keras.Sequential()

    for gene in chromosome[::-1]:
        decoder.add(gene_to_conv2d_transpose(gene))

    decoder.add(tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding='same', activation='sigmoid'))

    return decoder


def gene_to_conv2d(gene):
    return tf.keras.layers.Conv2D(filters=gene["filters"],
                                  kernel_size=(gene["kernel_size_x"], gene["kernel_size_y"]),
                                  strides=(gene["strides_x"], gene["strides_y"]),
                                  padding='same',
                                  activation='relu')


def gene_to_conv2d_transpose(gene):
    return tf.keras.layers.Conv2DTranspose(filters=gene["filters"],
                                           kernel_size=(gene["kernel_size_x"], gene["kernel_size_y"]),
                                           strides=(gene["strides_x"], gene["strides_y"]),
                                           padding='same',
                                           activation='relu')
