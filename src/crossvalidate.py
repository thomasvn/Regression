import numpy as np
import itertools
import math

class crossvalidate:
    def __init__(self, num_folds=5, initial_lambda=400):
        self.l = initial_lambda
        self.k = num_folds
        self.combos = self.create_combinations()
        self.rmse = [0] * 10

    def create_combinations(self):
        combos = itertools.combinations(range(self.k), int(self.k * 0.8))
        list_of_combos = []
        for i in combos:
            list_of_combos.append(list(i))
        return list_of_combos

    def determine_lambda(self, data):
        data = np.array_split(data, self.k)  # Split array into k folds

        # for index in range(10):
        #     rmse = 0
        #     for combo in self.combos:
        #         for fold in combo:  # Train with these folds

        #     # Find the average RMSE then log it to self.rmse
        #     self.l /= 2
