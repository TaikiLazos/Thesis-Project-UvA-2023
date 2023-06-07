import pandas as pd
import numpy as np

def load_x_y(path : str, n: int, m: int, normalize:bool = False):
    data = pd.read_csv(path, index_col='Date')['Adj Close']

    # We make x as an array of the last n day and y as the close price
    # This means we lose n rows of our data
    
    if normalize:
        data = (data-data.min())/(data.max()-data.min())

    x = [window.to_list() for window in data.rolling(window=n)][n - 1:-n]
    y = [window.to_list() for window in data.rolling(window=m)][n + m - 1 : -abs(n - m)]


    return x, y


def load_x_y_combined(path1: str, path2: str, n: int, m: int, normalize: bool = False):
    """
    path1: the company dataset
    path2: the commodity dataset
    n: the timeframe
    """
    company_data = pd.read_csv(path1, index_col='Date')['Adj Close']
    commodity_data = pd.read_csv(path2, index_col='Date')['Adj Close']

    # Normalize if needed
    if normalize:
        company_data = (company_data-company_data.min())/(company_data.max()-company_data.min())
        commodity_data = (commodity_data-commodity_data.min())/(commodity_data.max()-commodity_data.min())

    # We make x as an array of the last n day and y as the close price
    # This means we lose n rows of our data

    x1 = [window.to_list() for window in company_data.rolling(window=n)][n - 1:-n]
    x2 = [window.to_list() for window in commodity_data.rolling(window=n)][n - 1:-n]
    c = min(len(x1), len(x2))
    x = [[x1[-i], x2[-i]] for i in range(1, c + 1)][::-1]

    y1 = [window.to_list() for window in company_data.rolling(window=m)][n + m - 1:-abs(n - m)]
    y2 = [window.to_list() for window in commodity_data.rolling(window=m)][n + m - 1:-abs(n - m)]
    c = min(len(y1), len(y2))
    y = [[y1[-i], y2[-i]] for i in range(1, c + 1)][::-1]

    return x, y


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
        tmp = data_array[i:i+N]
        x = np.append(x, tmp)
    x = np.reshape(x, (samples, N, data_array.shape[1]))

    y = np.array([])
    samples = data_array.shape[0] - M + 1
    for i in range(samples):
        tmp = data_array[i:i+M]
        y = np.append(y, tmp)
    y = np.reshape(y, (samples, M, data_array.shape[1]))
    
    return x[:-M], y[N:]

