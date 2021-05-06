import matplotlib.pyplot as plt
import tensorflow as tf

import Dataset
import Evolution


if __name__ == "__main__":
    dataset = Dataset.Dataset()

    evolution = Evolution.Evolution()
    evolution.init_population()
    evolution.evaluate_population()
    print(f"Original population: {evolution.population}")
    print(f"Original evaluation: {evolution.evaluation}")

    generation_count = 5
    for generation in range(generation_count):
        recombinate_count = 2
        offsprings = []
        for recombinate_index in range(recombinate_count):
            parent1, parent2 = evolution.get_parents()
            offspring1, offspring2 = evolution.recombinate(parent1, parent2)
            offsprings.append(offspring1)
            offsprings.append(offspring2)

        mutated = Evolution.mutate(offsprings)
        evolution.combine_generations(mutated)
        evolution.set_new_population()

    autoencoder = evolution.get_best_individual()

    autoencoder.fit(dataset.train_noisy, dataset.train, epochs=10, shuffle=True,
                    validation_data=(dataset.test_noisy, dataset.test))

    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

    encoding_size = autoencoder.encoder.output.type_spec.shape[1] * \
        autoencoder.encoder.output.type_spec.shape[2] * \
        autoencoder.encoder.output.type_spec.shape[3]
    print(f"Size when encoded: {encoding_size}.")

    encoded_images = autoencoder.encoder(dataset.test).numpy()
    decoded_images = autoencoder.decoder(encoded_images).numpy()

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original + noise
        ax = plt.subplot(2, n, i + 1)
        plt.title("original + noise")
        plt.imshow(tf.squeeze(dataset.test_noisy[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        bx = plt.subplot(2, n, i + n + 1)
        plt.title("reconstructed")
        plt.imshow(tf.squeeze(decoded_images[i]))
        plt.gray()
        bx.get_xaxis().set_visible(False)
        bx.get_yaxis().set_visible(False)
    plt.show()
