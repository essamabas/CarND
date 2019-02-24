import numpy as np
import math

# TODO: Set weight1, weight2, and bias
weight1 = 4
weight2 = 5
bias = -9


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


test_inputs=[(1,1),(2,4),(5,-5),(-4,5)]

# Generate and check output
for test_input in test_inputs:
    # calculate 
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = sigmoid(linear_combination)
    print("output=", output)
