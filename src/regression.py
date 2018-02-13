import numpy as np

############################################################
#          4 Different Regression Implementations
############################################################
class regression:
    def __init__(self):
        pass

    def closed_form_lin_regression(self, training_data, training_results):
        # Calculate for weights matrix (96 rows x 1 column)
        weights = np.linalg.inv(training_data.transpose() * training_data) * \
            training_data.transpose() * training_results

        return weights


    def closed_form_ridge_regression(self):
        pass
        # Calculate for lambda by doing the following:
            # Cross Validate (Split into 5 pieces and choose 4 for training)
            # Try 10 different lambdas (400,200,100,50,25,12.5, ...)
            # Record the average error for the test piece
            # Shuffle the different pieces for training

        # Choose


    def gradient_descent_lin_regression(self):
        pass
        # Get 'x' and 'y' matrix as seen earlier

        # W^(t+1) = W^(t) + alpha * Xtranspose * (Y - X * Wtranspose)
        # Loop until || W^(t+1) - W^(t) || < epsilon (10^-6)

    def gradient_descent_ridge_regression(self):
        pass
        # Get lambda as seen earlier

        # W^(t+1) = W^(t) + alpha * (Xtranspose * (Y - X * Wtranspose) - lambda * Wtranspose)

