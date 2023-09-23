# Prisoner's Dilemma
# 2 people are caught and not allowed to communicate
# if both confess they get 3 yrs each
# if one testifies against other 0 yrs, 5 yrs
# if neither confesses nor testifies 1, 1 yrs
import os
import numpy as np
import random
import neat
import pickle
import importlib
import visualize as visualize

# values
ALGORITHM_FOLDER = "train_strategy"

n_generations = 100  # number of generations

n_history = 5  # nr of previous turns to judge on?
# match this to config file num_inputs

# length of game in training
GAME_LENGTH = 70

# loading config file
path = os.path.dirname(__file__)
file_name = '../../config/config.txt'
config_file = os.path.join(path, file_name)

# cooperate = 1, defect = 0, -1 means no previous history
C = 1
D = 0

# store histories and other misc info as needed
history = {}


# return score based on actions
def score(my_action, foe_action):
    if my_action == C and foe_action == C:
        return 3, 3
    if my_action == C and foe_action == D:
        return -1, 5
    if my_action == D and foe_action == C:
        return 5, -1
    if my_action == D and foe_action == D:
        return 0, 0


# trains against other neat individuals
def single_pairing(id1, f, genomes, nets):
    id2, genome2 = genomes[f - 1]
    history1 = history[id1]
    history2 = history[id2]

    sum1 = 0
    sum2 = 0

    for i in range(GAME_LENGTH):
        # run opposing player's prior actions through network and decide action to take
        output1 = nets[id1].activate(history2[-n_history:])
        action1 = np.argmax(output1)
        history[id1].append(action1)

        output2 = nets[id2].activate(history1[-n_history:])
        action2 = np.argmax(output2)
        history[id2].append(action2)

        # get score
        score1, score2 = score(action1, action2)
        sum1 += score1
        sum2 += score2
    return sum1, sum2


def get_visible_history(history, player, turn):
    history_so_far = history[:, :turn].copy()
    if player == 1:
        history_so_far = np.flip(history_so_far, 0)
    return history_so_far


# trains with traditional strategies
def neat_vs_standard_alg(id1, strategy, genomes, nets):
    func_A = importlib.import_module(ALGORITHM_FOLDER + "." + strategy)
    mem_A = None

    history = np.zeros((2, GAME_LENGTH), dtype=int)

    sum1 = 0
    sum2 = 0

    for i in range(GAME_LENGTH):
        # run opposing player's prior actions through network and decide action to take
        output1 = nets[id1].activate(history[0][-n_history:])
        action1 = np.argmax(output1)

        playerA_move, mem_A = func_A.get_action(get_visible_history(history, 0, i), mem_A)

        history[0, i] = playerA_move
        history[1, i] = action1

        score1, score2 = score(action1, playerA_move)
        sum1 += score1
        sum2 += score2
    return sum1


# tt against another bot
def eval_genomes(genomes, config):
    # reset
    nets = {}

    ids = []
    for genome_id, genome in genomes:
        genome.fitness = 0.
        nets[genome_id] = neat.nn.RecurrentNetwork.create(genome, config)

        history[genome_id] = [-1] * n_history
        ids.append(genome_id)

    # randomize order in player vs foe
    random.shuffle(ids)

    score_keeper = {}
    strategies = []
    # retrieve strategies to play against
    for file in os.listdir(ALGORITHM_FOLDER):
        if file.endswith(".py"):
            strategies.append(file[:-3])

    for strategy in strategies:
        score_keeper[strategy] = 0

    # play a round
    for genome_id, genome in genomes:
        # train_against_neat(genomes, genome, genome_id, nets)
        train_for_tournament(strategies, genomes, genome, genome_id, nets)

    # update fitness
    for genome_id, genome in genomes:
        totalScore = genome.fitness
        redefinedScore = tournament_score(totalScore, strategies)
        # redefinedScore = score_against_neat(totalScore, genomes)
        genome.fitness = redefinedScore


# updates fitness when training for tournament
def tournament_score(totalScore, strategies):
    return totalScore / (GAME_LENGTH * (len(strategies)))


# updates fitness when training against itself
def score_against_neat(totalScore, genomes):
    return -1/(totalScore / (GAME_LENGTH * (2 * len(genomes) - 2) * 3))


# method to keep score in games played
def train_for_tournament(strategies, genomes, genome, genome_id, nets):
    for strategy in strategies:
        my_score = neat_vs_standard_alg(genome_id, strategy, genomes, nets)
        genome.fitness += my_score


# method to keep score in games played against other neet algorithms
def train_against_neat(genomes, genome, genome_id, nets):
    for f in range(len(genomes)):
        # get score
        my_score, foe_score = single_pairing(genome_id, f + 1, genomes, nets)

        genome.fitness += my_score
        genomes[f][1].fitness += foe_score


def run():
    # load NEAT config file
    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_file)

    # populate algorithm
    p = neat.Population(config)

    # save statistics from each generation
    p.add_reporter(neat.StdOutReporter(False))
    stat = neat.StatisticsReporter()
    p.add_reporter(stat)

    # run for n_generations if not run until
    # desired fitness_threshold in config file
    winner = p.run(eval_genomes, n_generations)
    with open("../../EA/NEAT_algorithm.pickle.pkl", "wb") as f:
        pickle.dump(winner, f)
        f.close()
    # display winner genome
    print('\nBest genome:\n{!s}'.format(winner))

    # show output of winner against test data
    print('\nTest Output, Actual, Diff:')
    winner_network = neat.nn.RecurrentNetwork.create(winner, config)

    test_x = [[]]
    for i in range(5):
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

    predicted = []
    for xi in test_x:
        output = winner_network.activate(xi)
        action = np.argmax(output)
        predicted.append(action)

    print('----------   Winner vs test data   ----------')
    actions = ['Confess', 'Stay Silent']
    inputs = ['Unknown', 'Confess', 'Silent']

    for i in range(len(test_x)):
        hist = test_x[i]
        in_0 = hist[0] + 1
        in_1 = hist[1] + 1
        in_2 = hist[2] + 1
        in_3 = hist[3] + 1
        in_4 = hist[4] + 1
        print('History: %s %s %s %s %s -----> Action: %s  ' % (inputs[in_0], inputs[in_1], inputs[in_2],
                                                               inputs[in_3], inputs[in_4], actions[predicted[i]]))

    node_names = {-1: 'In -4', -2: 'In  -3', -3: 'In 2', -4: 'In -1', -5: 'In  -0', 0: 'Confess', 1: 'Silent'}
    # makes graphs for generation evolution and fitness
    visualize.plot_pop_fitness(stat, ylog=False, view=True)
    visualize.plot_species_change(stat, view=True)


# runs and trains the NEAT model
run()
