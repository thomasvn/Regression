import regression
import math


############################################################
#     Make predictions based on estimated parameters
############################################################
class prediction:
    def __init__(self):
        pass

    def linear_regression(self, features, responses, weights):
        # The predictions matrix will be (1 row x #instances columns)
        predictions = weights.transpose() * features.transpose()
        predictions = predictions.transpose()  # (#instances rows x 1 column)

        # Calculate for the Root Mean Square Error
        rmse = 0
        for i, prediction in enumerate(predictions):
            rmse += ((predictions[i] - responses[i]) ** 2)

        rmse = math.sqrt(rmse / len(predictions))
        
        return rmse