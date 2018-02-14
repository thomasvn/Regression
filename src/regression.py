import numpy as np
import crossvalidate

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
        pass
        cv = crossvalidate.crossvalidate()
        cv.determine_lambda(training_data)


    def gradient_descent_lin_regression(self):
        pass
        # Get 'x' and 'y' matrix as seen earlier

        # W^(t+1) = W^(t) + alpha * Xtranspose * (Y - X * Wtranspose)
        # Loop until || W^(t+1) - W^(t) || < epsilon (10^-6)

    def gradient_descent_ridge_regression(self):
        pass
        # Get lambda as seen earlier

        # W^(t+1) = W^(t) + alpha * (Xtranspose * (Y - X * Wtranspose) - lambda * Wtranspose)

