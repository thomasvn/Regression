import cleaning
import regression

############################################################
#                         Variables
############################################################

# Format of matrix used for training
#   [[-0.45 -1.85 -1.06 ...  1.26 -0.39  1.  ]
#    [-0.45 -0.27 -0.22 ... -0.62 -0.39  1.  ]
#    [-0.14  1.87  0.55 ...  0.52 -0.39  1.  ]
#    ... 
#    [ 0.81 -0.57 -0.48 ...  0.08  3.4   1.  ]
#    [ 0.18  0.28  1.   ...  0.73  0.52  1.  ]
#    [ 1.12  1.93  0.49 ... -0.49  3.77  1.  ]]
training_data_matrix = None  # (1595 rows x 96 columns)
testing_data_matrix = None  # (399 rows x 96 columns)

# Format of results matrix used for training and testing
#   [[0.67]
#    [0.43]
#    [0.12]
#    ...
#    [0.23]
#    [0.19]
#    [0.48]]
training_results_matrix = None  # (1595 rows x 1 column)
testing_results_matrix = None  # (399 rows x 1 column)


############################################################
#                      Driver Program
############################################################
c = cleaning.cleaning()

training_data_matrix = c.read_features_to_matrix('../data/crime-train.txt')
testing_data_matrix = c.read_ground_truth_to_matrix('../data/crime-train.txt')

r = regression.regression()

r.closed_form_lin_regression(training_data_matrix, testing_data_matrix)
