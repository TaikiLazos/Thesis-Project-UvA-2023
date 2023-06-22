# Create 180 days test set for period 1
# Load correct(N, M) model

import util
import os
import pandas as pd
import create_features
from keras.models import load_model
import data_process
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Create (N, M) dataset
N = 5
M = 10
VER = "without"
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

### WITHOUT COMMODITY DATASET ###
### Creating training dataset and test dataset
for commodity, cps in data_pair.items():
    path = f"Data/{commodity}"
    util.make_folder(f"Data/Test/{commodity}/n={N}&m={M}")
    for i, cp in enumerate(cps):
        # in case dataset does not exist
        if not os.path.exists(f"{path}/raw_data_{cp}.csv"):
            print(f"Dataset for {cp} was not found.")
            continue
        # load the company data
        df = pd.read_csv(f"{path}/raw_data_{cp}.csv")
        # Add features
        df = create_features.add_technical_indicators(df)

        # Add nominal label
        df["Company"] = i

        # Select the needed columns
        df = df[
            [
                "Date",
                "Adj Close",
                "Volume",
                "RSI",
                "MFI",
                "EMA",
                "SO",
                "MACD",
                "Company",
            ]
        ]
        # Normalize them (execpt for Date)
        target_cols = ["Adj Close", "Volume", "RSI", "MFI", "EMA", "SO", "MACD"]
        df[target_cols] = (df[target_cols] - df[target_cols].min()) / (
            df[target_cols].max() - df[target_cols].min()
        )

        # Drop cells with nan and inf
        df = df.dropna()

        target_cols = target_cols + ["Company"]

        # Take the latest 180 days as testing set and the rest for training + validation
        df_test = df.iloc[-360:-180]
        df_test.to_csv(
            f"Data/Test/{commodity}/n={N}&m={M}/x_without_{cp}_Strong.csv", index=True
        )

### WITH COMMODITY DATASET ###
### Creating training dataset and test dataset

for commodity, cps in data_pair.items():
    path = f"Data/{commodity}"
    util.make_folder(f"Data/Test/{commodity}/n={N}&m={M}")
    commodity_df = pd.read_csv(
        f"Data/{commodity}/raw_data_{commodity}.csv", index_col="Date"
    )
    for i, cp in enumerate(cps):
        # in case dataset does not exist
        if not os.path.exists(f"{path}/raw_data_{cp}.csv"):
            print(f"Dataset for {cp} was not found.")
            continue
        # load the company data
        df = pd.read_csv(f"{path}/raw_data_{cp}.csv")
        # Add commodity prices
        df = create_features.add_commodity(df, commodity_df)
        # Add features
        df = create_features.add_technical_indicators(df)

        # Add nominal label
        df["Company"] = i

        # Select the needed columns
        df = df[
            [
                "Date",
                "Adj Close",
                "Volume",
                "Adj Close Commodity",
                "RSI",
                "MFI",
                "EMA",
                "SO",
                "MACD",
                "Company",
            ]
        ]
        # Normalize them (execpt for Date)
        target_cols = [
            "Adj Close",
            "Volume",
            "Adj Close Commodity",
            "RSI",
            "MFI",
            "EMA",
            "SO",
            "MACD",
        ]
        df[target_cols] = (df[target_cols] - df[target_cols].min()) / (
            df[target_cols].max() - df[target_cols].min()
        )

        # Drop cells with nan and inf
        df = df.dropna()

        target_cols = target_cols + ["Company"]

        # Take the latest 180 days as testing set and the rest for training + validation
        df_test = df.iloc[-360:-180]
        df_test.to_csv(
            f"Data/Test/{commodity}/n={N}&m={M}/x_with_{cp}_Strong.csv", index=True
        )


# Load the Model
model = load_model(f"Model/model_N={N}_M={M}_{VER}")

# Load test data
RMSE = {}

for commodity, companies in data_pair.items():
    util.make_folder(f"Results/{commodity}/n={N}&m={M}")
    for company in companies:
        test_df = pd.read_csv(
            f"Data/Test/{commodity}/n={N}&m={M}/x_{VER}_{company}_Strong.csv"
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

        np.save(f"Results/{commodity}/n={N}&m={M}/y_true_{VER}_{company}_Strong", y_true)
        np.save(
            f"Results/{commodity}/n={N}&m={M}/y_predict_{VER}_{company}_Strong", prediction
        )
        np.save(
            f"Results/{commodity}/n={N}&m={M}/y_forecast_{VER}_{company}_Strong", forecast
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


def plot_results_CIA(commodity, company, N, M, VER):
    """
    It plots three lines in one plot to show the result
    """
    util.make_folder(f"Results/Plots/n={N}&m={M}")
    oil = pd.read_csv(f"Data/{commodity}/raw_data_{commodity}.csv")
    y_true = np.load(f"Results/{commodity}/n={N}&m={M}/y_true_{VER}_{company}_Strong.npy")
    y_predict = np.load(
        f"Results/{commodity}/n={N}&m={M}/y_predict_{VER}_{company}_Strong.npy"
    )
    y_forecast = np.load(
        f"Results/{commodity}/n={N}&m={M}/y_forecast_{VER}_{company}_Strong.npy"
    )
    oil["Adj Close"] = (oil["Adj Close"] - oil["Adj Close"].min()) / (
        oil["Adj Close"].max() - oil["Adj Close"].min()
    )
    oil = oil["Adj Close"].iloc[-360:- (360 - y_true.shape[0])]
    if y_true.shape != y_predict.shape and y_true.shape != y_forecast.shape:
        print("Something went wrong...\nYou don't have the same shape for the results.")
        return
    days = np.array(np.arange(len(y_true)))
    plt.plot(days, oil, label="Crude Oil")
    plt.plot(days, y_true, label=company)
    plt.plot(days, y_predict, label="One Step Forecasting")
    plt.plot(days, y_forecast, label="Multi Step Forecasting")
    plt.legend()
    plt.savefig(
        f"Results/Plots/n={N}&m={M}/result of {company}, N = {N}, M = {M}, VER = {VER} Strong Period.png"
    )
    plt.clf()

    # company = [
    #     "2222.SR",
    #     "XOM",
    #     "SHEL",
    #     "TTE",
    #     "CVX",
    #     "BP",
    #     "MPC",
    #     "VLO",
    # ]
    # VER = ["with" ,"without"]
    # for x in range(1, 6):
    #     for v in VER:
    #         for cp in company:
    #             plot_results_CIA("CL=F", cp, N*x, M*x, v)