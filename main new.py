# 1. Preprocessing -> preprocessing.py (Adds features and reformats in desirable numpy array)
# 2. Differential Privacy -> LS-LSTM shows improvement so why not -> main.py
# 3. Algorithm X -> I need to do something here like TDM & TDC equivalent. -> main.py
# 4. LSTM -> LSTM(32) -> dropout -> LSTM(16) -> dropout -> reformat to (M, # features) -> main.py
# 5. Prediction & Forecasting

import numpy as np
# import tensorflow_privacy
import tensorflow as tf
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import pandas as pd
import data_process
from sklearn.metrics import mean_squared_error

def lstm(x, y):
    """
    A LSTM model.
    It has 6 Keras layers, 
    LSTM(32) -> Dropout -> LSTM(18) -> Dropout -> Dense -> Reformat
    """
    # parameter setting
    epochs = 50
    batch_size = 32

    # # For differntial privacy
    # l2_norm_clip = 1.5
    # noise_multiplier = 1.3
    # num_microbatches = 250
    # learning_rate = 0.25

    # if batch_size % num_microbatches != 0:
    #     raise ValueError('Batch size should be an integer multiple of the number of microbatches')

    # optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
    #     l2_norm_clip=l2_norm_clip,
    #     noise_multiplier=noise_multiplier,
    #     num_microbatches=num_microbatches,
    #     learning_rate=learning_rate)

    # loss = tf.keras.losses.CategoricalCrossentropy(
    #     from_logits=True, reduction=tf.losses.Reduction.NONE)
    
    # building a model
    my_model = Sequential(
        [
            tf.keras.layers.LSTM(
                32,
                input_shape=x.shape[-2:], # (N, # of features)
                return_sequences=True,
            ),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.LSTM(18, return_sequences=True),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(x.shape[-1])),
            tf.keras.layers.Lambda(lambda x: x[:, -M:, :])  # Adjust output shape
        ]
    )


    # my_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    my_model.compile(optimizer="RMSprop", loss="mean_squared_error")

    my_model.summary()

    # Shuffle the data?

    # fit the model
    hist = my_model.fit(
        x, y, validation_split=0.15, epochs=epochs, batch_size=batch_size, verbose=0
    )

    return hist, my_model

def plot_loss_curve(h):
    """
    Plotting loss curve of a model.
    """
    plt.plot(h.history["loss"])
    plt.plot(h.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "valid"], loc="upper left")
    plt.show()


if __name__ == "__main__":
    # Global settings
    TRAIN = True # Train model or not?
    N = 14 # historical input window size N = 5, 7, 10, 14
    M = 3 # historical output window size, only 3 is available right now
    VER = "without" # with -> with commodity feature, without -> without commodity feature

    if TRAIN:
        # Load the training data
        x_train = np.load(f"Data/Training/n={N}/x_{VER}.npy", allow_pickle=True)
        y_train = np.load(f"Data/Training/n={N}/y_{VER}.npy", allow_pickle=True)

        print("Training Data Loading...")
        print(f"x = {x_train.shape}, y = {y_train.shape}")
        
        # Train the model
        history, model = lstm(x_train, y_train)

        # if you want you can plot the loss curve
        # plot_loss_curve(history)

        # Save the model
        model.save(f"Model/model_{N}_{VER}")
    else:
        model = load_model(f"Model/model_{N}_{VER}")
        model.summary()

    # Prediction
    # Load test data
    data_pair = {
        "CL=F": [
            "2222.SR",
            # "SNPMF",
            # "PCCYF",
            "XOM",
            "SHEL",
            "TTE",
            "CVX",
            "BP",
            "MPC",
            "VLO",
        ]
    }

    RMSE = {}

    for commodity, companies in data_pair.items():
        for company in companies:
            test_df = pd.read_csv(f"Data/Test/{commodity}/n={N}/x_{VER}_{company}.csv")
            tdf = test_df.iloc[:, 2:]
            x_test, y_test = data_process.create_x_y(tdf, N, M)
            y_true = y_test[::2, 0, 0]

            # Predicition result
            prediction = model.predict(x_test)

            # Forecasting result
            forecast = np.array([])
            xi = np.reshape(x_test[0], (1, N, x_test.shape[-1]))

            for _ in range(len(x_test)):
                yi = model.predict(xi)
                forecast = np.append(forecast, yi)
                # create a new xi
                xi = np.delete(xi, 0, 1)
                xi = np.append(xi, yi[:, 0:1, :], 1)
            
            # reshape the results
            forecast = np.reshape(forecast, prediction.shape)
            prediction = prediction[::2, 0, 0]
            forecast = forecast[::2, 0, 0]

            np.save(f"Results/{commodity}/N={N}/y_true_{VER}_{company}", y_true)
            np.save(f"Results/{commodity}/N={N}/y_predict_{VER}_{company}_{VER}", prediction)
            np.save(f"Results/{commodity}/N={N}/y_forecast_{VER}_{company}", forecast)

            # calculate the RMSE score and then save them
            s1 = mean_squared_error(y_true, prediction, squared=False)
            s2 = mean_squared_error(y_true, forecast, squared=False)

            RMSE[f'{commodity},{company}'] = [s1, s2]


    print(f"N = {N}, M = {M}, version = {VER}")
    print(RMSE)


    # If you get an error above run this. It gives a result of single company

    # company = "TTE"

    # test_df = pd.read_csv(f"Data/Test/CL=F/n={N}/x_{VER}_{company}.csv")
    # tdf = test_df.iloc[:, 2:]
    # x_test, y_test = data_process.create_x_y(tdf, N, M)

    # y_true = y_test[::2, 0, 0]

    # # Predicition result
    # prediction = model.predict(x_test)
    # prediction = prediction[::2, 0, 0]

    # # Forecasting result
    # forecast = np.array([])
    # xi = np.reshape(x_test[0], (1, N, x_test.shape[-1]))

    # for _ in range(len(x_test)):
    #     yi = model.predict(xi)
    #     forecast = np.append(forecast, yi)
    #     # create a new xi
    #     xi = np.delete(xi, 0, 1)
    #     xi = np.append(xi, yi[:, 0:1, :], 1)
    
    # # reshape the results
    # forecast = np.reshape(forecast, y_test.shape)
    # forecast = forecast[::2, 0, 0]

    # s1 = mean_squared_error(y_true, prediction, squared=False)
    # s2 = mean_squared_error(y_true, forecast, squared=False)

    # print(company)
    # print('RMSE of prediciton:', s1)
    # print('RMSE of forecasting', s2)

    # days = np.array(np.arange(len(y_true)))
    # plt.plot(days,y_true, label='true')
    # plt.plot(days, prediction, label='prediction')
    # plt.plot(days, forecast, label='forecasting')
    # plt.legend()
    # plt.savefig(f'Results/result of {company}, N = {N}, {VER}.png')