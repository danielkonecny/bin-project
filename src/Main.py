"""Autoencoder Design
Design Autoencoder for image denoising using Evolutionary Algorithm (Genetic Algorithm).
Course: Bio-Inspired Computers (BIN)
Organisation: Brno University of Technology - Faculty of Information Technologies
Author: Daniel Konecny (xkonec75)
File: Main.py
Version: 1.0
Date: 07. 05. 2021
"""


from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

import Autoencoder
import Dataset
import Evolution
import Transformer


def evolve_model(evolve_population_size=10, evolve_recombinate_count=5, evolve_generation_count=10):
    evolution = Evolution.Evolution()
    evolution.init_population(evolve_population_size)
    evolution.evaluate_population()
    print(f"Original population: {evolution.population}")
    print(f"Original evaluation: {evolution.evaluation}")

    for generation in range(evolve_generation_count):
        offsprings = []
        for recombinate_index in range(evolve_recombinate_count):
            parent1, parent2 = evolution.get_parents()
            offspring1, offspring2 = evolution.recombinate(parent1, parent2)
            size1 = Evolution.get_model_encoding(offspring1)
            if size1 < 32 * 32:
                print(f"New individual obtained by recombination with size {size1} - {offspring1}.")
                offsprings.append(offspring1)
            size2 = Evolution.get_model_encoding(offspring2)
            if size2 < 32 * 32:
                print(f"New individual obtained by recombination with size {size2} - {offspring2}.")
                offsprings.append(offspring2)

        mutated = Evolution.mutate(offsprings)
        evolution.combine_generations(mutated)
        evolution.set_new_population()

    evolved_autoencoder = evolution.get_best_individual()

    return evolved_autoencoder


def load_model():
    model = [{'filters': 16, 'kernel_size_x': 4, 'kernel_size_y': 5, 'strides_x': 1, 'strides_y': 2},
             {'filters': 15, 'kernel_size_x': 1, 'kernel_size_y': 2, 'strides_x': 1, 'strides_y': 1},
             {'filters': 3, 'kernel_size_x': 3, 'kernel_size_y': 3, 'strides_x': 2, 'strides_y': 1}]

    encoder = Transformer.chromosome_to_encoder(model)
    decoder = Transformer.chromosome_to_decoder(model)
    loaded_autoencoder = Autoencoder.Autoencoder(encoder, decoder)
    loaded_autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    return loaded_autoencoder


def model_preview_and_save(trained_autoencoder):
    encoding_size = trained_autoencoder.encoder.output.type_spec.shape[1] * \
        trained_autoencoder.encoder.output.type_spec.shape[2] * \
        trained_autoencoder.encoder.output.type_spec.shape[3]
    print(f"Size when encoded: {encoding_size}.")

    trained_autoencoder.encoder.summary()
    trained_autoencoder.decoder.summary()

    Path("model").mkdir(parents=True, exist_ok=True)
    trained_autoencoder.save('model/autoencoder')


def model_load_and_preview():
    loaded_autoencoder = tf.keras.models.load_model('model/autoencoder')

    encoding_size = loaded_autoencoder.encoder.output.type_spec.shape[1] * \
        loaded_autoencoder.encoder.output.type_spec.shape[2] * \
        loaded_autoencoder.encoder.output.type_spec.shape[3]
    print(f"Size when encoded: {encoding_size}.")

    loaded_autoencoder.encoder.summary()
    loaded_autoencoder.decoder.summary()
    return loaded_autoencoder


def plot_results(plot_autoencoder, plot_dataset):
    encoded_images = plot_autoencoder.encoder(plot_dataset.test).numpy()
    decoded_images = plot_autoencoder.decoder(encoded_images).numpy()

    n = 10
    plt.figure(figsize=(22, 8))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.title("original")
        plt.imshow(tf.squeeze(plot_dataset.test[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display original + noise
        ax = plt.subplot(3, n, i + n + 1)
        plt.title("original + noise")
        plt.imshow(tf.squeeze(plot_dataset.test_noisy[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        bx = plt.subplot(3, n, i + 2 * n + 1)
        plt.title("reconstructed")
        plt.imshow(tf.squeeze(decoded_images[i]))
        plt.gray()
        bx.get_xaxis().set_visible(False)
        bx.get_yaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":
    """ Modes of the app:
    0 - Find the best model via evolution, train it and test it.
    1 - Load already found model, train it and test it.
    2 - Load already trained model and test it.
    """
    mode = 0

    population_size = 40
    recombinate_count = 10
    generation_count = 5

    dataset = Dataset.Dataset()

    if mode == 0:
        autoencoder = evolve_model(population_size, recombinate_count, generation_count)
    elif mode == 1:
        autoencoder = load_model()
    else:
        autoencoder = model_load_and_preview()

    if mode == 0 or mode == 1:
        autoencoder.fit(dataset.train_noisy,
                        dataset.train,
                        epochs=10,
                        shuffle=True,
                        validation_data=(dataset.test_noisy, dataset.test))
        model_preview_and_save(autoencoder)

    plot_results(autoencoder, dataset)
