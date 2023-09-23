from __future__ import division

import os
import pickle
import random
import torch as torch
import torch.nn as nn
import math as math
import numpy as np
import random as rm
import itertools

import importlib


class Particle:
    def __init__(self, mem_len):
        self.fitness = 2  # indidual fitness
        self.history = []  # History of opponents move
        self.mem_len = mem_len  # Length of history
        self.sigmoids = 2  # Number of sigmoids stored
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

    # Gets the value inside a single sigmoid
    def get_inner_sum(self, sigmoid):
        inner_sum = self.inner_constants[sigmoid]
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
        inner_sum = self.unknown_constant
        for i in range(self.mem_len - len(self.history)):
            inner_sum += self.unknown_inner[i]
        outer_sum = torch.tensor(inner_sum)
        outer_sum = (torch.sigmoid(outer_sum) * 2 - 1) * self.unknown_outer
        return outer_sum

    # Selects strategy based upon known history
    def strategy(self):
        summ = 0
        # Get the value of the sigmoids
        for i in range(self.sigmoids):
            inner_sum = torch.tensor(self.get_inner_sum(i))
            summ += (torch.sigmoid(inner_sum) * 2 - 1) * self.outer_coeffs[i]
        # Get the value of the unknown sigmoid
        summ += self.get_unknown_sum()
        if summ > 0:
            return 1
        else:
            return 0
