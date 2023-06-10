# 1. Preprocessing -> preprocessing.py (Adds features and reformats in desirable numpy array)
# 2. Differential Privacy -> LS-LSTM shows improvement so why not -> main.py
# 3. Algorithm X -> I need to do something here like TDM & TDC equivalent. -> main.py
# 4. LSTM -> LSTM(32) -> dropout -> LSTM(16) -> dropout -> reformat to (M, # features) -> main.py
# 5. Prediction & Forecasting 

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, load_model
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import os


def univariate_model(X_train, y_train):
    """
    create single layer rnn model trained on X_train and y_train
    and make predictions on the X_test data
    """

    # create a model
    model = Sequential(
        [
            tf.keras.layers.LSTM(32, input_shape=X_train.shape[-2:]),
            tf.keras.layers.Dense(m),
        ]
    )

    model.compile(optimizer="adam", loss="mean_squared_error")

    # fit the model
    history = model.fit(
        X_train, y_train, validation_split=0.15, epochs=50, batch_size=64, verbose=0
    )

    return history, model


def multivariate_model(X_train, y_train):
    """
    create single layer rnn model trained on X_train and y_train
    and make predictions on the X_test data
    """

    # create a model
    model = Sequential(
        [
            tf.keras.layers.LSTM(
                32,
                input_shape=X_train.shape[-2:],
                return_sequences=True,
            ),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(X_train.shape[-1])),
            tf.keras.layers.Lambda(lambda x: x[:, -y_train.shape[1]:, :])  # Adjust output shape
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")

    model.summary()

    # fit the model
    history = model.fit(
        X_train, y_train, validation_split=0.15, epochs=50, batch_size=64, verbose=0
    )

    return history, model


def plot_loss_curve(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "valid"], loc="upper left")
    plt.show()


def RMSE(y_true, y_hat):
    return mean_squared_error(y_true, y_hat, squared=False)


if __name__ == "__main__":
    # Set True if you want to train
    TRAIN = False

    # This sets the historical window size
    n = 5
    m = 3
    ver = "with"  # with -> with commodity feature (multivariate model), without -> (univariate model)

    if TRAIN:
        # the data is normalized, and has everything except the latest 20% SHEL data
        x = np.load(f"Data/Training/n={n}/x_{ver}.npy")
        y = np.load(f"Data/Training/n={n}/y_{ver}.npy")


        if ver == "with":
            x = np.array([i.T for i in x])
            y = np.array([i.T for i in y])
            print("Training data shape")
            print(x.shape, y.shape)

            history, my_model = multivariate_model(x, y)
        else:
            # reshape
            x = np.reshape(x, (x.shape[0], x.shape[1], 1))
            print("Training data shape")
            print(x.shape, y.shape)

            history, my_model = univariate_model(x, y)

        my_model.save(f"Model/LSTM_model_n={n}_{ver}")
        plot_loss_curve(history)
    else:
        my_model = load_model(f"Model/LSTM_model_n={n}_{ver}")

    my_model.summary()

    ### Forecasting target company ###
    target = "SHEL"

    # Loading Test dataset of the target company
    X_test = np.load(f"Data/Test/n={n}/x_{ver}_{target}.npy")
    Y_test = np.load(f"Data/Test/n=5/y_without_{target}.npy")

    # Forecasting is for predicting future trend and Prediction is for predicting short term
    forecast = np.array([])
    X_test = np.array([i.T for i in X_test])
    predict = my_model.predict(X_test)
    print(X_test[0])
    print(predict[0])

    # Save the results
    commodity = "CL=F"
    if not os.path.exists(f"Data/Results/Prediction/{commodity}"):
        os.makedirs(f"Data/Results/Prediction/{commodity}")
    np.save(f"Data/Results/Prediction/{commodity}/{ver}_n={n}", predict)

    if ver == 'with':
        X_test = np.array([i.T for i in X_test])
        Y_test = np.array([i.T for i in Y_test])
        xi = np.reshape(X_test[0], (1, n, 2))

        for _ in tqdm(range(len(X_test))):
            yi = np.mean(my_model.predict(xi), axis=1)
            forecast = np.append(forecast, yi[0][0])
            # pop the first index and append the yi to the last index
            xi =  np.delete(xi, [0], 1)
            xi = np.append(xi[0], yi, 0)
            xi = np.reshape(xi, (1, n, 2))
    else:
        X_test = np.reshape(X_test, (30, n, 1))
        xi = np.reshape(X_test[0], (1, n))

        for _ in tqdm(range(len(X_test))):
            yi = np.mean(my_model.predict(np.reshape(xi, (1,n,1))))
            forecast = np.append(forecast, yi)
            # pop the first index and append the yi to the last index
            xi =  np.delete(xi, [0], 1)
            yi = np.reshape(yi, (1, 1))
            xi = np.append(xi, yi, 1)

    # Save the results
    commodity = 'CL=F'
    if not os.path.exists(f"Data/Results/{commodity}"):
        os.makedirs(f"Data/Results/{commodity}")
    np.save(f"Data/Results/{commodity}/{ver}_n={n}", forecast)
