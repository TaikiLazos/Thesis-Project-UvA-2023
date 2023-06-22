# 1. Preprocessing -> preprocessing.py (Adds features and reformats in desirable numpy array)
# 2. Differential Privacy -> LS-LSTM shows improvement so why not -> main.py
# 3. Algorithm X -> I need to do something here like TDM & TDC equivalent. -> main.py
# 4. LSTM -> LSTM(32) -> dropout -> LSTM(16) -> dropout -> reformat to (M, # features) -> main.py
# 5. Prediction & Forecasting

import os
import numpy as np
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import pandas as pd
import data_process
import util
import lstm_model
import preprocessing

if __name__ == "__main__":
    ### Global settings ##########################################################

    # Load the desired data
    # N is the input window size and M is the output window size
    # Possible (N, M) combinations are:
    #  (N, M) = (2, 1), (4, 2), (6, 3), (8, 4), (10, 5)
    #  (N, M) = (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)
    #  (N, M) = (1, 2), (2, 4), (3, 6), (4, 8), (5, 10)
    N = 5
    M = 10

    # Specify the model
    # "with" -> with the commodity feature, "without" -> without the commodity feature
    VER = "with"

    data_pair = {
        "CL=F": [
            "2222.SR",
            "XOM",
            "SHEL",
            "TTE",
            "CVX",
            "BP",
            "MPC",
            "VLO",
        ]
    }

    #############################################################################

    ### Loading Data and Training a LSTM model ##################################

    if not os.path.exists(f"Model/model_N={N}_M={M}_{VER}"):
        # If there is no training data then you need to run preprocessing.py
        if not os.path.exists(f"Data/Training/n={N}&m={M}/x_{VER}.npy"):
            print("No Data found.")
            print(f"Creating dataset for N = {N}, M = {M}, VER = {VER}...\n")
            preprocessing.preprocessing(N=N, M=M, data_pair=data_pair, GET_RAW=False) # Set GET_RAW False after your initial run

        # Load the training data
        print("Training Data Loading...")
        x_train = np.load(f"Data/Training/n={N}&m={M}/x_{VER}.npy", allow_pickle=True)
        y_train = np.load(f"Data/Training/n={N}&m={M}/y_{VER}.npy", allow_pickle=True)
        print(f"x = {x_train.shape}, y = {y_train.shape}\n")

        # Train the model
        history, model = lstm_model.lstm(x_train, y_train, M)

        # if you want you can plot the loss curve
        # lstm_model.plot_loss_curve(history)

        # Save the model
        util.make_folder("Model")
        model.save(f"Model/model_N={N}_M={M}_{VER}")
    else:
        model = load_model(f"Model/model_N={N}_M={M}_{VER}")
        model.summary()

    #############################################################################

    ### Prediction ##############################################################

    # Load test data
    RMSE = {}

    for commodity, companies in data_pair.items():
        util.make_folder(f"Results/{commodity}/n={N}&m={M}")
        for company in companies:
            test_df = pd.read_csv(
                f"Data/Test/{commodity}/n={N}&m={M}/x_{VER}_{company}.csv"
            )
            # Exclude the index and Date columns -> test_df.iloc[:, 2:]
            x_test, y_test = data_process.create_x_y(test_df.iloc[:, 2:], N, M)
            num_features = y_test.shape[2]

            # Collect all ground truth close prices
            y_true = np.append(y_test[:-1, 0, 0], y_test[-1, :, 0])

            # Predicition result
            prediction = model.predict(x_test)
            prediction = np.append(prediction[:-1, -1, 0], np.flip(y_test[-1, :, 0]))

            # Forecasting result
            forecast = np.array([])
            xi = np.reshape(x_test[0], (1, N, x_test.shape[-1]))

            # for _ in range(len(x_test)):
            #     yi = model.predict(xi)
            #     forecast = np.append(forecast, yi)
            #     # create a new xi
            #     xi = np.delete(xi, 0, 1)
            #     xi = np.append(xi, yi[:, -1:, :], 1)
            
            while forecast.shape[0] < y_true.shape[0]:
                yi = model.predict(xi)
                pprice = yi[:, :, 0]
                forecast = np.append(forecast, pprice)
                # update xi
                if N <= M:
                    xi = yi[:, :N, :]
                else:
                    xi = np.append(xi[:, (N - M):, :], yi[:, :, :])
                    xi = np.reshape(xi, (1, N, num_features))
                    

            # reshape the results
            forecast = forecast[:y_true.shape[0]]

            np.save(f"Results/{commodity}/n={N}&m={M}/y_true_{VER}_{company}", y_true)
            np.save(
                f"Results/{commodity}/n={N}&m={M}/y_predict_{VER}_{company}", prediction
            )
            np.save(
                f"Results/{commodity}/n={N}&m={M}/y_forecast_{VER}_{company}", forecast
            )

            # calculate the RMSE score and then save them
            s1 = mean_squared_error(y_true, prediction, squared=False)
            s2 = mean_squared_error(y_true, forecast, squared=False)

            RMSE[f"{commodity},{company}"] = [s1, s2]

    print(f"N = {N}, M = {M}, version = {VER}")
    print(
        f"RMSE for Single Forecasting: {sum(val[0] for val in RMSE.values()) / len(RMSE)}, RMSE for Multistep Forecasting: {sum(val[1] for val in RMSE.values()) / len(RMSE)}"
    )
    print("RMSE for each company (Single Forecasting, Multistep Forecasting):\n", RMSE)
    print("-------All Work Run Succesfully-------")
