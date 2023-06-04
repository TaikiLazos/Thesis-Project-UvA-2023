import pandas as pd
import numpy as np

# "./Data/exxon_data.csv"
# path = "./Data/exxon_data.csv"
# n = 3

def load_x_y(path : str, n: int, normalize:bool = False):
    data = pd.read_csv(path, index_col='Date')['Adj Close']

    # We make x as an array of the last n day and y as the close price
    # This means we lose n rows of our data
    
    if normalize:
        data = (data-data.min())/(data.max()-data.min())

    x = [window.to_list() for window in data.rolling(window=n)][n - 1:-n]
    y = [window.to_list() for window in data.rolling(window=n)][2*n - 1:]

    return x, y


def load_x_y_combined(path1: str, path2: str, n: int, normalize: bool = False):
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

    y1 = [window.to_list() for window in company_data.rolling(window=n)][2*n - 1:]
    y2 = [window.to_list() for window in commodity_data.rolling(window=n)][2*n - 1:]
    c = min(len(y1), len(y2))
    y = [[y1[-i], y2[-i]] for i in range(1, c + 1)][::-1]

    return x, y


def ts_train_test(x, y):
    '''
    input: 
        x, y: x values and y values 
    output:
      X_train, y_train: 80% of the data
      X_test: 20% of the data
    '''
    # create training and test set
    split = 0.8

    X_train = np.array((x[:int(len(x) * split)]))
    Y_train  = np.array(y[:int(len(x) * split)])
    X_test = np.array(x[int(len(x) * split):])
    Y_test = np.array(y[int(len(x) * split):])


    # reshape
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    return X_train, Y_train , X_test, Y_test