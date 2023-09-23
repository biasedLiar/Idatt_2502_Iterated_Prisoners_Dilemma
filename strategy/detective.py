import random
import numpy as np


# The detective will spend the first four steps testing out the opponent.
# It will always C, D, C, D as it's first steps
# It then looks at how the opponent has responded.
# If the opponent always cooperated, it will defect for the rest of the game
# If the opponent defected at least once, it will impliment tit-for-tat for the rest of the game


def get_action(history, mem):
    history_test = [1, 0, 1, 0]
    l = history.shape[1]
    trick = mem
    decision = None

    if l < 4:
        decision = history_test[l]
    elif l == 4:
        opponent_move = history[1]
        if np.count_nonzero(opponent_move - 1) == 0:
            trick = True
        else:
            trick = False

    if l >= 4:
        if trick:
            decision = 0
        else:
            decision = history[1, -1]

    return decision, trick
