# 1. Download raw data from yfinance (save on disk)
# 2. Add features (commodity and/or technical indicators depending on the model)
# 3. Normalize the necesary columns
# 4. Split the data. Save them individually on local drive
# 5. Concatnate everything that is not used as test set
# 6. Select the necessary columns and reshape them to a numpy array
#    -> This will be used for training and validation
# 7. Save the numpy array

import yfinance as yf
import util
import os
import pandas as pd
import create_features
import data_process
import numpy as np


def preprocessing(
    N, M, data_pair, GET_RAW=False, WITHOUT_COMMODITY=True, WITH_COMMODITY=True
):
    """
    This function does all the preprocessing steps
    """
    # Global variables
    START_DATE = "2000-08-23"  # This is the oldest date yfinance has
    END_DATE = "2023-05-01"

    # Data Pairs Commodity - Related Company
    if len(data_pair) == 0:
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

    ### SAVING RAW DATA INTO LOCAL FOLDER ###
    if GET_RAW:
        # make sure there is a folder called Data
        util.make_folder("Data")
        # writing raw data into each folder
        for commodity, cps in data_pair.items():
            # make a folder for that commodity if that does not exists
            util.make_folder(f"Data/{commodity}")
            # download commodity dataset
            raw_data = yf.download(commodity, START_DATE, END_DATE)
            raw_data.to_csv(f"Data/{commodity}/raw_data_{commodity}.csv", index=True)
            # do that for companies
            for cp in cps:
                raw_data = yf.download(cp, START_DATE, END_DATE)
                raw_data.to_csv(f"Data/{commodity}/raw_data_{cp}.csv", index=True)

    # Randomly select five different dates for test data
    # test_periods = data_process.choose_5_periods(f"Data/CL=F")
    # print(test_periods)

    ### WITHOUT COMMODITY DATASET ###
    if WITHOUT_COMMODITY:
        ### Creating training dataset and test dataset
        util.make_folder("Data/Test")
        x, y = np.array([]), np.array([])

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
                df_test = df.tail(180)
                df_training = df.iloc[:-180]
                df_test.to_csv(
                    f"Data/Test/{commodity}/n={N}&m={M}/x_without_{cp}.csv", index=True
                )
                x_train, y_train = data_process.create_x_y(
                    df_training[target_cols], N, M
                )
                x = np.append(x, x_train)
                y = np.append(y, y_train)

        # Reshape
        samples = int(len(x) / (N * len(target_cols)))
        x = np.reshape(x, (samples, N, len(target_cols)))
        y = np.reshape(y, (samples, M, len(target_cols)))

        # Save the data
        if x.shape[0] == y.shape[0]:
            print(f"N = {N}, M = {M}")
            print("Without commodity dataset was succesfully created.")
            util.make_folder(f"Data/Training/n={N}&m={M}")
            np.save(f"Data/Training/n={N}&m={M}/x_without", x)
            np.save(f"Data/Training/n={N}&m={M}/y_without", y)

    ### WITH COMMODITY DATASET ###
    if WITH_COMMODITY:
        ### Creating training dataset and test dataset
        util.make_folder("Data/Test")
        x, y = np.array([]), np.array([])

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
                df_test = df.tail(180)
                df_training = df.iloc[:-180]
                df_test.to_csv(
                    f"Data/Test/{commodity}/n={N}&m={M}/x_with_{cp}.csv", index=True
                )
                x_train, y_train = data_process.create_x_y(
                    df_training[target_cols], N, M
                )
                x = np.append(x, x_train)
                y = np.append(y, y_train)

        # Reshape
        samples = int(len(x) / (N * len(target_cols)))
        x = np.reshape(x, (samples, N, len(target_cols)))
        y = np.reshape(y, (samples, M, len(target_cols)))

        # Save the data
        if x.shape[0] == y.shape[0]:
            print(f"N = {N}, M = {M}")
            print("With commodity dataset was succesfully created.")
            util.make_folder(f"Data/Training/n={N}&m={M}")
            np.save(f"Data/Training/n={N}&m={M}/x_with", x)
            np.save(f"Data/Training/n={N}&m={M}/y_with", y)
