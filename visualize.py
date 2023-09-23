from __future__ import print_function

import copy
import warnings
import seaborn as sns
import graphviz
import matplotlib.pyplot as plt
import numpy as np


def plot_pop_fitness(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    # Plots the population's average and best fitness.
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())

    plt.plot(generation, avg_fitness, 'b-', label="average fitness")
    plt.plot(generation, best_fitness, 'r-', label="best fitness")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_species_change(statistics, view=False, filename='Species_Evolution.svg'):
    # Visualizes speciation throughout evolution.
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    col = sns.color_palette("Paired")
    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves, cmap='Blues')

    plt.title("Species Evolution")
    plt.ylabel("Size of Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()