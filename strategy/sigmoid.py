import os
import pickle as pickle
import neat
import random
import numpy as np
import torch
import random as rm
from particle import Particle


# -------------Less trained version that somehow does better---------------------
def get_action(history, winner_genome):
    if winner_genome is None:
        winner_genome = replay_genome()
    his_len = winner_genome.mem_len
    history_2 = [-1] * his_len
    if len(history[1]) < his_len:
        history_2 = get_history_so_far(history[1], len(history[0]), his_len)
    else:
        for i in range(his_len):
            history_2[i] = history[1][len(history[1]) - his_len + i]

    winner_genome.history = history_2
    output = winner_genome.strategy()
    return output, winner_genome


def replay_genome():
    with open('sigmoid/Sigmoid.pickle.pkl', "rb") as f:
        genome = pickle.load(f)

    return genome


def get_history_so_far(history, turn, mem_len):
    new_hist = []
    if turn >= mem_len:
        for i in range(mem_len):
            new_hist.append(history[turn - mem_len + i])
    else:
        for i in range(turn):
            new_hist.append(history[i])
    return new_hist
