import os
import pickle
import neat
import random
import numpy as np


def get_action(history, winner_network):

    if winner_network is None:
        winner_network = replay_genome()
    his_len = 5
    history_2 = [-1] * his_len
    if len(history[0]) < his_len:
        diff = his_len - 1 - len(history[1])

        for i in range(len(history[0])):
            history_2[i + diff] = history[1][i]
    else:
        for i in range(his_len):

            history_2[i] = history[1][len(history[0]) - his_len + i]
    output = winner_network.activate(history_2)
    action = np.argmax(output)

    return action, winner_network


def replay_genome():
    path = os.path.dirname(__file__)
    file_name = '../config/config.txt'
    config_file = os.path.join(path, file_name)
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_file)
    with open('EA/NEAT_tournament.pickle.pkl', "rb") as f:
        genome = pickle.load(f)

    return neat.nn.RecurrentNetwork.create(genome, config)