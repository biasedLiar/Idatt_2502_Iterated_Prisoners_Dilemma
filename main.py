import os
import itertools
import importlib
import numpy as np
import random
from particle import Particle

ALGORITHM_FOLDER = "strategy"
RESULT_FILE = "results.txt"

point_array = [[0, 5], [-1, 3]]
labels = ["D", "C"]


BENCH_MARKING = True
ALG_NUM = 0

# Gets the history so far, oriented for the active player.
def get_visible_history(history, player, turn):
    history_so_far = history[:, :turn].copy()
    if player == 1:
        history_so_far = np.flip(history_so_far, 0)
    return history_so_far

# runs a round of IPD with a pair of strategies
def run_round(pair):
    func_A = importlib.import_module(ALGORITHM_FOLDER + "." + pair[0])
    func_B = importlib.import_module(ALGORITHM_FOLDER + "." + pair[1])
    mem_A = None
    mem_B = None

    LENGTH_OF_GAME = int(200 - 40 * np.log(1 - random.random()))
    history = np.zeros((2, LENGTH_OF_GAME), dtype=int)

    for turn in range(LENGTH_OF_GAME):
        playerA_move, mem_A = func_A.get_action(get_visible_history(history, 0, turn), mem_A)
        playerB_move, mem_B = func_B.get_action(get_visible_history(history, 1, turn), mem_B)
        history[0, turn] = playerA_move
        history[1, turn] = playerB_move

    return history

# Gets the score of a round of IPD
def get_scores(history):
    player_A_score = 0
    player_B_score = 0
    ROUND_LENGTH = history.shape[1]
    for turn in range(ROUND_LENGTH):
        player_A_move = history[0, turn]
        player_B_move = history[1, turn]
        player_A_score += point_array[player_A_move][player_B_move]
        player_B_score += point_array[player_B_move][player_A_move]
    return player_A_score / ROUND_LENGTH, player_B_score / ROUND_LENGTH


# Writes the result of an round to file
def write_output_results(f, pair, history, scoresA, scoresB):
    f.write(pair[0] + " (P1)  VS.  " + pair[1] + " (P2)\n")
    for p in range(2):
        for t in range(history.shape[1]):
            move = history[p, t]
            f.write(labels[move] + " ")
        f.write("\n")
    f.write("Final score for " + pair[0] + ": " + str(scoresA) + "\n")
    f.write("Final score for " + pair[1] + ": " + str(scoresB) + "\n")
    f.write("\n")


def pad(my_string, num_char):
    out = my_string
    for i in range(len(my_string), num_char):
        out = out + " "
    return out

# Runs the tournament
def run_tournament(algorithm_folder, outFile):
    print("Tournament starting, reading files from: " + algorithm_folder)
    score_keeper = {}
    strategies = []
    # Gets the names of the files to be read
    for file in os.listdir(algorithm_folder):
        if file.endswith(".py"):
            strategies.append(file[:-3])


    # Sets scores to 0
    for strategy in strategies:
        score_keeper[strategy] = 0


    # Prepares file to be written out
    f = open(outFile, "w+")
    # Runs a IPD round for each pair of strategies
    for pair in itertools.combinations(strategies, r=2):
        # Runs the round
        round_ = run_round(pair)
        # Gets the scores of the final round
        player_A_score, player_B_score = get_scores(round_)
        # Writes the result to output file
        write_output_results(f, pair, round_, player_A_score, player_B_score)
        #Updates scores
        score_keeper[pair[0]] += player_A_score
        score_keeper[pair[1]] += player_B_score

    # Creates and sorts score list
    scores = np.zeros(len(score_keeper))
    for i in range(len(strategies)):
        scores[i] = score_keeper[strategies[i]]
    placement = np.argsort(scores)

    # Writes scores to file
    f.write("\n\nSTRATEGY SCORES\n")
    for res in range(len(strategies)):
        i = placement[-1 - res]
        score = scores[i]
        avg_score = score / (len(strategies) - 1)
        f.write("#" + str(res + 1) + ": " + pad(strategies[i] + ":",
                                                 16) + ' %.3f' % score + '  (%.3f' % avg_score + " average)\n")

    f.flush()
    f.close()
    print("Tournament finished. Output written to: " + RESULT_FILE)


if __name__ == "__main__":
    run_tournament(ALGORITHM_FOLDER, RESULT_FILE)