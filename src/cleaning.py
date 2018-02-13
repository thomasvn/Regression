import numpy as np

############################################################
#        Parsing Testing & Training Data for NumPy
############################################################
class cleaning:
    def __init__(self):
        pass


    def read_features_to_matrix(self, data_file):
        array = np.loadtxt(data_file, delimiter='\t', skiprows=1, 
            usecols=range(1,96))
        
        # Add column of '1's as the intercept
        if 'train' in data_file:
            intercept = np.ones((1595,1))
        else:
            intercept = np.ones((399,1))
        array = np.append(array, intercept, axis=1)

        matrix = np.matrix(array)
        return matrix


    def read_ground_truth_to_matrix(self, data_file):
        array = np.loadtxt(data_file, delimiter='\t', skiprows=1, usecols=0)
        matrix = np.matrix(array).transpose()
        return matrix