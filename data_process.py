import pandas as pd
import numpy as np
import os


def create_x_y(df, N, M):
    """
    Make x and y from a pd.Dataframe
    x => (# of samples, N, # of features)
    y => (# of samples, M, # of features)
    """
    data_array = df.values
    x = np.array([])
    samples = data_array.shape[0] - N + 1

    for i in range(samples):
        tmp = data_array[i : i + N]
        x = np.append(x, tmp)
    x = np.reshape(x, (samples, N, data_array.shape[1]))

    y = np.array([])
    samples = data_array.shape[0] - M + 1
    for i in range(samples):
        tmp = data_array[i : i + M]
        y = np.append(y, tmp)
    y = np.reshape(y, (samples, M, data_array.shape[1]))

    return x[:-M], y[N:]


def choose_5_periods(path):
    """
    Given a path to a folder with csv files, I select the shortest dataframe and
    choose 5 random time slots. This will be returned as a list of Dates.
    """
    # Load the folder
    if not os.path.exists(path):
        print("Somethign went wrong with creating test sets.")
        return []

    # Find the shortest dataframe
    shortest_length = float("inf")
    shortest_file = ""

    for file_name in os.listdir(path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(path, file_name)

            # Read the CSV file and calculate the number of rows
            df = pd.read_csv(file_path)
            data_length = len(df)

            # Check if the current file has the shortest data length so far
            if data_length < shortest_length:
                shortest_length = data_length
                shortest_file = file_name

    # Select 5 Dates from the shortest_file
    df = pd.read_csv(os.path.join(path, shortest_file))

    dates = df["Date"].sample(n=5)

    return dates.to_list()
