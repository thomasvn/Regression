import numpy as np
import itertools
import math
import regression
import prediction

class crossvalidate:
    def __init__(self, num_folds=5, initial_lambda=400):
        self.l = initial_lambda
        self.k = num_folds
        self.combos = self.create_combinations()
        self.rmse = {}


    def create_combinations(self):
        combos = itertools.combinations(range(self.k), int(self.k * 0.8))
        list_of_combos = []
        for i in combos:
            list_of_combos.append(list(i))
        return list_of_combos


    def determine_lambda(self, data, results):
        reg = regression.regression()
        pred = prediction.prediction()

        data = np.array_split(data, self.k)  # Split array into k folds
        results = np.array_split(results, self.k)

        for index in range(10):  # Iterate through all lambdas
            rmse = 0
            for combo in self.combos:  # Find avg RMSE from all combinations
                data_matrices_to_stack = []
                results_matrices_to_stack = []
                for fold_index in combo:
                    data_matrices_to_stack.append(data[fold_index])
                    results_matrices_to_stack.append(results[fold_index])
                training_data = np.vstack(tuple(data_matrices_to_stack))
                results_data = np.vstack(tuple(results_matrices_to_stack))
                weights = reg.closed_form_lin_regression(training_data, results_data, l=self.l)
                single_rmse = pred.linear_regression(training_data, results_data, weights)
                rmse += single_rmse

            # Calculate the avg RMSE for all combos and log with lambda value
            rmse /= len(self.combos)
            self.rmse[self.l] = rmse

            # Divide lambda by 2 then try combinations again
            self.l = float(self.l / 2)

        # Return lambda with smallest average RMSE
        return min(self.rmse, key=self.rmse.get)
