# --- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division

import os
import pickle
import torch as torch
import math as math
import numpy as np
import random as rm
import matplotlib.pyplot as plt

import importlib

# --- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)
D = 0
C = 1
LETTER_LIST = ["D", "C"]
# Odds of mutation happening on any one gene
MUTATE_CHANCE = 0.1
# Odds of a more fit indidual getting priority when reproducing
MERITOCRACY = 0.2


MEMORY_LENGTH = 4
SIGMOIDS = 1
ALGORITHM_FOLDER = "train_strategy"

#V1/V2
# V1: False
# V2+: True
TRAIN_AGAINST_STATIC = True

#V2/V3
# V2-: Softmax
# V3+: Sigmoid
ACT_SIGMOID = 0
ACT_SOFTMAX = 1
ACT_FUNC = ACT_SIGMOID

#V3/V4
# V3-: False
# V4+: True
USE_CONSTANT = True 

#V4/V5
# V4-: False
# V5: True
USE_UNKNOWN = True



# --- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self, mem_len):
        self.fitness = 2  # indidual fitness
        self.history = []  # History of opponents move
        self.mem_len = mem_len  # Length of history
        self.sigmoids = SIGMOIDS  # Number of sigmoids stored
        # Sigmoid coefficients
        self.outer_coeffs = [rm.uniform(-1, 1) for i in range(self.sigmoids)]
        self.inner_coeffs = [[rm.uniform(-1, 1) for i in range(mem_len)] for i in range(self.sigmoids)]
        self.inner_constants = [rm.uniform(-1, 1) for i in range(self.sigmoids)]
        # Unknown history coefficients
        self.unknown_inner = [rm.uniform(-1, 1) for i in range(self.mem_len)]
        self.unknown_outer = rm.uniform(-1, 1)
        self.unknown_constant = rm.uniform(-1, 1)

    # Resets memory
    def reset(self):
        self.history = [-1] * self.mem_len

    # Applies the correct activation function
    def act_func(self, value):
        input  = torch.tensor(value)
        if ACT_FUNC == ACT_SIGMOID:
            return torch.sigmoid(input)
        else:
            return torch.softmax(input, 0, dtype=float).item()

    # Gets the value inside a single sigmoid
    def get_inner_sum(self, sigmoid):
        inner_sum = 0
        if USE_CONSTANT:
            inner_sum += self.inner_constants[sigmoid]
        #inner_sum = self.inner_constants[sigmoid]
        # If history is short, use all of history
        if len(self.history) < self.mem_len:
            for i in range(len(self.history)):
                inner_sum += self.history[i] * self.inner_coeffs[sigmoid][self.mem_len + i - len(self.history)]
        # If history is long, only use mem_len of it
        else:
            for i in range(self.mem_len):
                inner_sum += self.history[i] * self.inner_coeffs[sigmoid][i]
        return inner_sum

    # Gets the value of the unknown sigmoid
    def get_unknown_sum(self):
        
        inner_sum = 0
        if USE_CONSTANT:
            inner_sum += self.unknown_constant
        
        for i in range(self.mem_len - len(self.history)):
            inner_sum += self.unknown_inner[i]
        outer_sum = (self.act_func(inner_sum) * 2 - 1) * self.unknown_outer
        #outer_sum = (torch.sigmoid(outer_sum) * 2 - 1) * self.unknown_outer
        return outer_sum

    # Selects strategy based upon known history
    def strategy(self):
        summ = 0
        # Get the value of the sigmoids
        for i in range(self.sigmoids):
            #summ += (self.act_func(self.get_inner_sum(i)) * 2 - 1) * self.outer_coeffs[i]
            summ += (self.act_func(self.get_inner_sum(i)) * 2 - 1) * self.outer_coeffs[i]
        # Get the value of the unknown sigmoid
        if USE_UNKNOWN:
            summ += self.get_unknown_sum()
        if summ > 0:
            return 1
        else:
            #print(summ)
            return 0


# Gets scores for both players based on actions
def get_score(my_action, foe_action):
    if my_action == C and foe_action == C:
        return 3, 3
    if my_action == C and foe_action == D:
        return -1, 5
    if my_action == D and foe_action == C:
        return 5, -1
    if my_action == D and foe_action == D:
        return 0, 0


# Randomly splits and concatenates to lists
def split_list(list1, list2):
    new_list = [0] * len(list1)
    rand = rm.uniform(0, len(list1))
    for i in range(0, math.floor(rand)):
        new_list[i] = list1[i]
    for i in range(math.floor(rand), len(list1)):
        new_list[i] = list2[i]
    return new_list


# Randomly returns 1 of 2 values
def return_random(val1, val2):
    rand = rm.uniform(0, 2)
    if (rand > 1):
        return val1
    return val2


# Creates a child based upon two parents
def create_child(parent1, parent2):
    child = Particle(parent1.mem_len)
    # Child randomly splits the values of it's two parents
    child.outer_coeffs = split_list(parent1.outer_coeffs, parent2.outer_coeffs)
    child.inner_constants = split_list(parent1.inner_constants, parent2.inner_constants)
    child.unknown_inner = split_list(parent1.unknown_inner, parent2.unknown_inner)
    child.unknown_constant = return_random(parent1.unknown_constant, parent2.unknown_constant)
    child.unknown_outer = return_random(parent1.unknown_outer, parent2.unknown_outer)
    for i in range(len(parent1.inner_coeffs)):
        child.inner_coeffs[i] = split_list(parent1.inner_coeffs[i], parent2.inner_coeffs[i])
    return child


# Randomly mutates 0 or more items in a list
def mutate_list(list):
    for i in range(len(list)):
        rand = rm.uniform(0, 1)
        if rand < MUTATE_CHANCE:
            rand = rm.uniform(0, 0.5)
            list[i] += rand - 0.25


# Randomly maybe mutates a constant
def mutate_const(const):
    rand = rm.uniform(0, 1)
    if rand < MUTATE_CHANCE:
        rand = rm.uniform(0, 0.5)
        const += rand - 0.25
    return const


# Mutates a child
def mutate_child(child):
    mutate_list(child.outer_coeffs)
    mutate_list(child.inner_constants)
    child.unknown_outer = mutate_const(child.unknown_outer)
    child.unknown_constant = mutate_const(child.unknown_constant)
    for i in range(len(child.inner_coeffs)):
        mutate_list(child.inner_coeffs[i])
    # Sometimes, the child experiences larger mutations
    if rm.uniform(0, 1) < MUTATE_CHANCE:
        mutate_child(child)


# Returns a random individual based upon fitness and random chance
def select_single_breeder(pop):
    # A random individual starts being the chosen one
    rand = rm.uniform(0, len(pop))
    chosen = pop[math.floor(rand)]
    for ind in pop:
        # For every individual with higher fitness, there is a chance that they will replace the chosen one
        if chosen.fitness < ind.fitness and rm.uniform(0, 1) < MERITOCRACY:
            chosen = ind
    return chosen


# Generates a child based upon the population
def get_child(pop):
    # Selects parents
    parent1 = select_single_breeder(pop)
    parent2 = select_single_breeder(pop)
    # Creates child from parents
    child = create_child(parent1, parent2)
    # Mutates child
    mutate_child(child)
    return child


# Gets entire history regardless of length.
def get_visible_history(history, player, turn):
    history_so_far = history[:, :turn].copy()
    if player == 1:
        history_so_far = np.flip(history_so_far, 0)
    return history_so_far


# Gets correct length of history with unknowns replaced with -1
def get_last_x_moves(history, turn, mem_len):
    new_hist = [-1] * mem_len
    if turn < mem_len:
        for i in range(turn):
            new_hist[mem_len - i - 1] = history[turn - i - 1]
    else:
        for i in range(mem_len):
            new_hist[mem_len - i - 1] = history[turn - i - 1]
    return new_hist


# Gets correct length of history with unknowns simply removed
def get_history_so_far(history, turn, mem_len):
    new_hist = []
    if turn >= mem_len:
        for i in range(mem_len):
            new_hist.append(history[turn - mem_len + i])
    else:
        for i in range(turn):
            new_hist.append(history[i])
    return new_hist


# Removes the -1 from the starting list of numbers
def start_turns_to_history(history):
    new_hist = []
    for i in range(len(history)):
        if history[i] != -1:
            new_hist.append(history[i])
    return new_hist


# Runs a single round of IPD between an individual and a classic strategy
def run_round(ind, strat):
    # Sets up the classic strategy
    func_B = importlib.import_module(ALGORITHM_FOLDER + "." + strat)
    mem_B = None

    # Sets length of game and history
    LENGTH_OF_GAME = 200
    history = np.zeros((2, LENGTH_OF_GAME), dtype=int)

    score1 = 0

    for turn in range(LENGTH_OF_GAME):
        # formatizes history
        ind.history = get_history_so_far(history[1], turn, ind.mem_len)
        # Gets strategy of the individual
        playerA_move = ind.strategy()
        # Gets strategy of the classic algorithm
        playerB_move, mem_B = func_B.get_action(get_visible_history(history, 1, turn), mem_B)
        # Updates history
        history[0, turn] = playerA_move
        history[1, turn] = playerB_move
        # Updates scores
        scores = get_score(playerA_move, playerB_move)
        score1 += scores[0]
    # Updates fitness
    score1 = score1 / LENGTH_OF_GAME
    ind.fitness += score1

    return history

def run_round_vs_other(ind1, ind2):
    # Sets length of game and history
    LENGTH_OF_GAME = 200
    history = np.zeros((2, LENGTH_OF_GAME), dtype=int)

    score1 = 0
    score2 = 0

    for turn in range(LENGTH_OF_GAME):
        # formatizes history
        ind1.history = get_history_so_far(history[1], turn, ind1.mem_len)
        # Gets strategy of the individual
        playerA_move = ind1.strategy()
        # formatizes history
        ind2.history = get_history_so_far(history[0], turn, ind1.mem_len)
        # Gets strategy of the individual
        playerB_move = ind2.strategy()
        # Updates history
        history[0, turn] = playerA_move
        history[1, turn] = playerB_move
        # Updates scores
        scores = get_score(playerA_move, playerB_move)
        score1 += scores[0]
        score2 += scores[1]
    # Updates fitness
    score1 = score1 / LENGTH_OF_GAME
    score2 = score2 / LENGTH_OF_GAME
    ind1.fitness += score1
    ind2.fitness += score2

    return history


# Runs a single round of IPD between an individual and a classic strategy and prints the result
def run_round_with_print(ind, strat):
    # Sets up the classic strategy
    func_B = importlib.import_module(ALGORITHM_FOLDER + "." + strat)
    mem_B = None

    # Sets length of game and history
    LENGTH_OF_GAME = 200
    history = np.zeros((2, LENGTH_OF_GAME), dtype=int)

    score1 = 0
    score2 = 0

    for turn in range(LENGTH_OF_GAME):
        # formatizes history
        ind.history = get_history_so_far(history[1], turn, ind.mem_len)
        # Gets strategy of the individual
        playerA_move = ind.strategy()
        # Gets strategy of the classic algorithm
        playerB_move, mem_B = func_B.get_action(get_visible_history(history, 1, turn), mem_B)
        # Updates history
        history[0, turn] = playerA_move
        history[1, turn] = playerB_move
        # Updates scores
        scores = get_score(playerA_move, playerB_move)
        score1 += scores[0]
        score2 += scores[1]
    # Updates fitness
    score1 = score1 / LENGTH_OF_GAME
    score2 = score2 / LENGTH_OF_GAME
    ind.fitness += score1

    # Prints round and results
    print("\n\nGenome vs ", strat)
    my_string = ""
    strat_string = ""
    for i in range(len(history[0])):
        my_string += LETTER_LIST[history[0][i]] + " "
        strat_string += LETTER_LIST[history[1][i]] + " "
    print("Genome:\n", my_string)
    print(strat, ":\n", strat_string)
    print("Genome score: ", score1)
    print(strat, " score: ", score2)
    return history, score1


# Calculates fitness by having every algorithm run against every traditional strategy
def evaluate_fitness(algorithm_folder, pop):
    # Gets strategies
    strategies = []
    for file in os.listdir(algorithm_folder):
        if file.endswith(".py"):
            strategies.append(file[:-3])
    # Each individual fights against each strategy
    for ind in pop:
        ind.fitness = 0
        for strategy in strategies:
            run_round(ind, strategy)
            # print(strategy, ": ", ind.fitness)
        ind.fitness = ind.fitness / len(strategies)
        print(ind.fitness)
   
# Calculates fitness by having every algorithm run against every traditional strategy
def evaluate_fitness_vs_others(pop):
    for i in range(len(pop)):
        pop[i].fitness = 0
    for i in range(len(pop)):
        for j in range(i + 1, len(pop)):
            run_round_vs_other(pop[i], pop[j])
            # print(strategy, ": ", ind.fitness)
        pop[i].fitness = pop[i].fitness / len(pop)
        print(pop[i].fitness)


# Evaluates fitness of a single algorithm by having it run against every traditional strategy.
# and prints the results.
def evaluate_single_fitness(algorithm_folder, ind):
    # Gets strategies
    strategies = []
    for file in os.listdir(algorithm_folder):
        if file.endswith(".py"):
            strategies.append(file[:-3])

    # The individual fights against each strategy and prints result
    ind.fitness = 0
    strat_scores = []
    for strategy in strategies:
        round, score = run_round_with_print(ind, strategy)
        strat_scores.append(score)
        # print(strategy, ": ", ind.fitness)
    ind.fitness = ind.fitness / len(strategies)
    print("\n\n")
    for i in range(len(strat_scores)):
        print("Score vs ", strategies[i], ": ", strat_scores[i])


# Returns the individual with the highest fitness
def get_strongest(pop):
    strongest = pop[0]
    for ind in pop:
        if ind.fitness > strongest.fitness:
            strongest = ind
    return strongest


# Returns the average fitness of the population
def get_average_fitness(pop):
    summ = 0.0
    for ind in pop:
        summ += ind.fitness + 0.0
    summ = summ / len(pop)
    return summ


# Runs the program
class EvAlg():
    def __init__(self, num_particles, maxiter, mem_len):
        # Create list to store best and average fitness throughout the generations.
        best = []
        average = []

        # establish the swarm
        swarm = []
        for i in range(0, num_particles):
            swarm.append(Particle(mem_len))

        # run through the generations
        i = 0
        while i < maxiter:
            print("Generation: ", i)
            # Check fitness
            evaluate_fitness(ALGORITHM_FOLDER, swarm)
            # Save best and average fitness
            best.append(get_strongest(swarm).fitness)
            average.append(get_average_fitness(swarm))

            if not TRAIN_AGAINST_STATIC:
                print("VS others")
                evaluate_fitness_vs_others(swarm)
            # Create next generation
            next_gen = []
            # The strongest gets saved
            next_gen.append(get_strongest(swarm))
            # Create the rest of the new generation
            for j in range(num_particles - 1):
                next_gen.append(get_child(swarm))
            swarm = next_gen
            i += 1

        # print final results
        print('FINAL:')

        # Create test input to see how the winner responds
        test_x = [[]]
        for i in range(mem_len):
            new_test_x = []
            for j in range(len(test_x)):
                temp_test = [item for item in test_x[j]]
                temp_test.append(0)
                new_test_x.append(temp_test)
                temp_test = [item for item in test_x[j]]
                temp_test.append(1)
                new_test_x.append(temp_test)
            temp_test = [-1] * (i + 1)
            new_test_x.append(temp_test)
            test_x = new_test_x

        # Do a final fitness check
        evaluate_fitness(ALGORITHM_FOLDER, swarm)
        for ind in swarm:
            print("Fitness:", ind.fitness)

        # Find and print strongest individual
        particle = get_strongest(swarm)
        print(particle.fitness)
        evaluate_single_fitness(ALGORITHM_FOLDER, particle)

        # See how the strongest reacts to different stimuli
        predicted = []
        for xi in test_x:
            particle.history = start_turns_to_history(xi)
            action = particle.strategy()
            predicted.append(action)

        actions = ['Confess', 'Stay Silent']
        inputs = ['Unknown', 'Confess', 'Silent']

        # Print the strongest reaction to the different stimuli
        print('----------   Winner test   ----------')
        for i in range(len(test_x)):
            hist = test_x[i]
            my_string = "History: "
            for j in range(mem_len):
                my_string += str(inputs[hist[j] + 1]) + " "
            my_string += "-----> Action: " + str(actions[predicted[i]])
            print(my_string)

        # Save the strongest individual
        with open("../../sigmoid/Sigmoid.pickle.pkl", "wb") as f:
            pickle.dump(particle, f)
            # pickle.dump(particle, f, -1)
            f.close()


        print("Best:")
        for i in range(len(best)):
            print(best[i])
        
        print("\n\nAverage:")
        for i in range(len(average)):
            print(average[i])

        # Plot average and best individuals throughout the generations
        plt.plot(range(maxiter), average, 'b-', label="average fitness")
        plt.plot(range(maxiter), best, 'r-', label="best fitness")

        plt.title("Population's average and best fitness")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.grid()
        plt.legend(loc="best")
        # Save the plot
        filename = 'avg_fitness.svg'
        plt.savefig(filename)

        # Show the plot
        plt.show()
        plt.close()


# --- RUN ----------------------------------------------------------------------+
# Run the program
# num_particles: population size
# maxiter: Number generations
EvAlg(num_particles=15, maxiter=20, mem_len=MEMORY_LENGTH)

# --- END ----------------------------------------------------------------------+
