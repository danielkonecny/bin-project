"""Autoencoder Design
Design Autoencoder for image denoising using Evolutionary Algorithm (Genetic Algorithm).
Course: Bio-Inspired Computers (BIN)
Organisation: Brno University of Technology - Faculty of Information Technologies
Author: Daniel Konecny (xkonec75)
File: Evolution.py
Date: 07. 05. 2021
"""


import copy
import numpy as np
import tensorflow as tf

import Autoencoder
import Dataset
import Transformer

chromosome_test = [{
    "filters": 16,
    "kernel_size_x": 3,
    "kernel_size_y": 3,
    "strides_x": 2,
    "strides_y": 2
}, {
    "filters": 8,
    "kernel_size_x": 3,
    "kernel_size_y": 3,
    "strides_x": 2,
    "strides_y": 1
}, {
    "filters": 4,
    "kernel_size_x": 1,
    "kernel_size_y": 1,
    "strides_x": 2,
    "strides_y": 2
}, {
    "filters": 2,
    "kernel_size_x": 3,
    "kernel_size_y": 3,
    "strides_x": 1,
    "strides_y": 1
}]

chromosome_original = [{
    "filters": 16,
    "kernel_size_x": 3,
    "kernel_size_y": 3,
    "strides_x": 2,
    "strides_y": 2
}, {
    "filters": 8,
    "kernel_size_x": 3,
    "kernel_size_y": 3,
    "strides_x": 2,
    "strides_y": 2
}]


def mutate(individuals):
    for individual_index in range(len(individuals)):
        mutated_individual = copy.deepcopy(individuals[individual_index])
        mutated = False

        print(f"Before mutation: {mutated_individual}")

        for gene_index in range(len(mutated_individual)):
            random = np.random.uniform(0, 1, 5)
            if random[0] < 0.2 and mutated_individual[gene_index]['filters'] <= 14:
                mutated_individual[gene_index]['filters'] += 2
                mutated = True
            elif random[0] > 0.8 and mutated_individual[gene_index]['filters'] >= 4:
                mutated_individual[gene_index]['filters'] -= 2
                mutated = True
            if random[1] < 0.05 and mutated_individual[gene_index]['kernel_size_x'] <= 5:
                mutated_individual[gene_index]['kernel_size_x'] += 1
                mutated = True
            elif random[1] > 0.95 and mutated_individual[gene_index]['kernel_size_x'] >= 2:
                mutated_individual[gene_index]['kernel_size_x'] -= 1
                mutated = True
            if random[2] < 0.05 and mutated_individual[gene_index]['kernel_size_y'] <= 5:
                mutated_individual[gene_index]['kernel_size_y'] += 1
                mutated = True
            elif random[2] > 0.95 and mutated_individual[gene_index]['kernel_size_y'] >= 2:
                mutated_individual[gene_index]['kernel_size_y'] -= 1
                mutated = True
            if random[3] < 0.05 and mutated_individual[gene_index]['strides_x'] == 1:
                mutated_individual[gene_index]['strides_x'] = 2
                mutated = True
            elif random[3] < 0.05 and mutated_individual[gene_index]['strides_x'] == 2:
                mutated_individual[gene_index]['strides_x'] = 1
                mutated = True
            if random[4] < 0.05 and mutated_individual[gene_index]['strides_y'] == 1:
                mutated_individual[gene_index]['strides_y'] = 2
                mutated = True
            elif random[4] < 0.05 and mutated_individual[gene_index]['strides_y'] == 2:
                mutated_individual[gene_index]['strides_y'] = 1
                mutated = True

        if mutated is True:
            print(f"Different after mutation: {mutated_individual}")
        else:
            print(f"Identical after mutation: {mutated_individual}")

        size = get_model_encoding(mutated_individual)
        if mutated and size < 32 * 32:
            individuals[individual_index] = copy.deepcopy(mutated_individual)
            # print(f"New individual obtained by mutation with size {size} - {individuals[individual_index]}.")

    return individuals


def get_model_encoding(chromosome):
    filters = chromosome[-1]['filters']
    output_shape_x = 32
    output_shape_y = 32

    for gene in chromosome:
        output_shape_x //= gene['strides_x']
        output_shape_y //= gene['strides_y']

    return filters * output_shape_x * output_shape_y


class Evolution:
    def __init__(self):
        self.population = []
        self.evaluation = []
        self.dataset = Dataset.Dataset()
        self.chromosome_length = 0
        self.population_size = 0

    def init_population(self, population_size=10, chromosome_length=3):
        self.chromosome_length = chromosome_length
        self.population_size = population_size

        population_index = 0
        while population_index < population_size:
            chromosome = []
            filters = np.random.randint(low=2, high=17, size=chromosome_length)
            kernel_size = np.random.randint(low=1, high=7, size=(2, chromosome_length))
            strides = np.random.randint(low=1, high=3, size=(2, chromosome_length))

            for chromosome_index in range(chromosome_length):
                chromosome.append({
                    "filters": filters[chromosome_index],
                    "kernel_size_x": kernel_size[0][chromosome_index],
                    "kernel_size_y": kernel_size[1][chromosome_index],
                    "strides_x": strides[0][chromosome_index],
                    "strides_y": strides[1][chromosome_index]
                })

            size = get_model_encoding(chromosome)
            if size < 32 * 32:
                print(f"New individual obtained by initialization with size {size} - {chromosome}.")
                self.population.append(chromosome)
                population_index += 1

    def evaluate_population(self):
        self.evaluation = []
        for individual in self.population:
            encoder = Transformer.chromosome_to_encoder(individual)
            decoder = Transformer.chromosome_to_decoder(individual)
            autoencoder = Autoencoder.Autoencoder(encoder, decoder)
            autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
            history = autoencoder.fit(self.dataset.train_noisy,
                                      self.dataset.train,
                                      epochs=1,
                                      shuffle=True,
                                      validation_data=(self.dataset.test_noisy, self.dataset.test))
            self.evaluation.append(history.history['val_loss'][-1])
        print(f"Evaluated: {self.evaluation}")

    def get_parents(self):
        inverted = np.reciprocal(self.evaluation)
        proportions = []
        accumulated = 0
        inverted_sum = np.sum(inverted)
        for fitness in inverted:
            accumulated += fitness / inverted_sum
            proportions.append(accumulated)

        chosen_index1 = 0
        random1 = np.random.uniform(0, 1)
        while True:
            if random1 < proportions[chosen_index1]:
                break
            chosen_index1 += 1

        chosen_index2 = 0
        random2 = (random1 + 0.5) % 1
        while True:
            if random2 < proportions[chosen_index2]:
                break
            chosen_index2 += 1

        print(f"Parent1: {self.population[chosen_index1]}")
        print(f"Parent2: {self.population[chosen_index2]}")

        return self.population[chosen_index1], self.population[chosen_index2]

    def recombinate(self, parent1, parent2):
        crossing_site = np.random.randint(low=1, high=self.chromosome_length)
        recombination1 = parent1[:crossing_site] + parent2[crossing_site:]
        recombination2 = parent2[:crossing_site] + parent1[crossing_site:]

        print(f"Offspring1: {recombination1}")
        print(f"Offspring2: {recombination2}")

        return recombination1, recombination2

    def combine_generations(self, offsprings):
        offsprings_eval = []
        for offspring in offsprings:
            encoder = Transformer.chromosome_to_encoder(offspring)
            decoder = Transformer.chromosome_to_decoder(offspring)
            autoencoder = Autoencoder.Autoencoder(encoder, decoder)
            autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
            history = autoencoder.fit(self.dataset.train_noisy,
                                      self.dataset.train,
                                      epochs=1,
                                      shuffle=True,
                                      validation_data=(self.dataset.test_noisy, self.dataset.test))
            offsprings_eval.append(history.history['val_loss'][-1])

        self.population += offsprings
        self.evaluation += offsprings_eval

    def set_new_population(self):
        print(f"Old population: {self.population}")
        print(f"Old evaluation: {self.evaluation}")
        sorted_population = [x for _, x in sorted(zip(self.evaluation, self.population))]
        self.population = sorted_population[:self.population_size]
        self.evaluation = (sorted(self.evaluation))[:self.population_size]
        print(f"New population: {self.population}")
        print(f"New evaluation: {self.evaluation}")

    def get_best_individual(self):
        sorted_population = [x for _, x in sorted(zip(self.evaluation, self.population))]

        encoder = Transformer.chromosome_to_encoder(sorted_population[0])
        decoder = Transformer.chromosome_to_decoder(sorted_population[0])
        autoencoder = Autoencoder.Autoencoder(encoder, decoder)
        autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
        return autoencoder
