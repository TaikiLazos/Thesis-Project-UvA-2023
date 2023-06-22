import os
import numpy as np

def make_folder(path):
    """
    If folder does not exist then make one.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def find_average_shifted_Window(array, M):
    """
    This function finds the average of a shifted window input array.
    """

    output_array = np.zeros((array.shape[0], array.shape[0] + M - 1))
    for i in range(array.shape[0]):
        for j in range(M):
            output_array[i][i + j] = array[i][j]

    # take the sum and find average for each day
    output_array = np.sum(output_array, axis = 0)

    # border problem
    for i in range(M - 1):
        output_array[i] = output_array[i] / (i + 1)
        output_array[-(i + 1)] = output_array[-(i + 1)] / (i + 1)

    output_array[M - 1: -M + 1] = output_array[M - 1: -M + 1] / M

    return output_array
