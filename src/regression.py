import numpy as np
import crossvalidate
import math

############################################################
#          4 Different Regression Implementations
############################################################
class regression:
    def __init__(self):
        pass


    def closed_form_lin_regression(self, training_data, training_results, l=0):
        # Calculate for weights matrix (96 rows x 1 column)
        weights = np.linalg.inv(training_data.transpose() * training_data + l * \
            np.identity(96)) * training_data.transpose() * training_results

        return weights


    def closed_form_ridge_regression(self, training_data, training_results):
        cv = crossvalidate.crossvalidate()
        l = cv.determine_lambda(training_data, training_results)
        weights = self.closed_form_lin_regression(training_data, training_results, l)

        return weights


    def gradient_descent_lin_regression(self, training_data, training_results, l=0):
        weights = np.random.normal(0,1,(96,1))  # Randomly generated gaussian weights
        alpha = 1 * (10 ** -2)
        epsilon = 1 * (10 ** -10)

        loss = self.loss_function(training_data, training_results, weights)

        while True:
            new_weights = weights + (((alpha * training_data.transpose()) * 
                (training_results - training_data * weights) - l * 
                weights) / 1595)

            # Compute values of Loss Function
            new_loss = self.loss_function(training_data, training_results, new_weights)

            # If loss is minimal, we can stop and use these weights
            if abs(new_loss - loss) < epsilon:
                break

            weights = new_weights
            loss = new_loss

        return new_weights


    def gradient_descent_ridge_regression(self, training_data, training_results):
        cv = crossvalidate.crossvalidate()
        l = cv.determine_lambda(training_data, training_results)
        weights = self.gradient_descent_lin_regression(training_data, 
            training_results, l)

        return weights


    def loss_function(self, training_data, training_results, weights, l=0):
        first_term = (training_data * weights - training_results)
        # second_term = l * weights.transpose() * np.identity(1) * weights
        return math.sqrt(first_term.transpose() * first_term) / 1596
