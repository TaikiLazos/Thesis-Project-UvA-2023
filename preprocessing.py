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

# Global variables
GET_RAW = False
WITHOUT_COMMODITY = True
WITH_COMMODITY = True
N = 5  # window size: 5, 7, 10, 14
M = 3
START_DATE = "2000-08-23"  # This is the oldest date yfinance has
END_DATE = "2023-05-01"

# Data Pairs Commodity - Related Company
data_pair = {
    "CL=F": [
        "2222.SR",
        "SNPMF",
        "PCCYF",
        "XOM",
        "SHEL",
        "TTE",
        "CVX",
        "BP",
        "MPC",
        "VLO",
    ]
}
#  'NG=F': ['RELIANCE.NS', 'VLO', 'PSX', 'MPC', 'BP', 'CVX', 'TTE'],
#  'GC=F': ['NEM', 'GOLD', 'AEM', 'AU', 'PLZL.IL', 'GFI', 'KGC', 'NCM', 'FCX'],
#  'SI=F': ['HL', 'CDE', 'SVL.AX', '0815.HK', 'NILI.V', 'WPM.TO'],
#  'KC=F': ['SBUX']}

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

### WITHOUT COMMODITY DATASET ###
if WITHOUT_COMMODITY:
    ### Creating training dataset and test dataset
    util.make_folder("Data/Test")
    x, y = np.array([]), np.array([])

    for commodity, cps in data_pair.items():
        path = f"Data/{commodity}"
        util.make_folder(f"Data/Test/{commodity}/n={N}")
        for cp in cps:
            # in case dataset does not exist
            if not os.path.exists(f"{path}/raw_data_{cp}.csv"):
                print(f"Dataset for {cp} was not found.")
                continue
            # load the company data
            df = pd.read_csv(f"{path}/raw_data_{cp}.csv")
            # Add features
            df = create_features.add_technical_indicators(df)
            
            # Select the needed columns
            df = df[['Date', 'Adj Close', 'Volume', 'RSI', 'MFI', 'EMA', 'SO', 'MACD']]
            # Normalize them (execpt for Date)
            target_cols = ['Adj Close', 'Volume', 'RSI', 'MFI', 'EMA', 'SO', 'MACD']
            df[target_cols] = (df[target_cols]-df[target_cols].min())/(df[target_cols].max()-df[target_cols].min())

            # Take the latest 60 days as testing set and the rest for training + validation
            df_test = df.tail(60)
            df_training = df.iloc[:-60]
            # x_test, y_test = data_process.create_x_y(df_test[target_cols], N, M)
            df_test.to_csv(f"Data/Test/{commodity}/n={N}/x_without_{cp}.csv", index=True)
            x_train, y_train = data_process.create_x_y(df_training[target_cols], N, M)
            x = np.append(x, x_train)
            y = np.append(y, y_train)
    
    # Reshape
    samples = int(len(x) / (N * len(target_cols)))
    x = np.reshape(x, (samples, N, len(target_cols)))
    y = np.reshape(y, (samples, M, len(target_cols)))
    
    # Save the data
    if x.shape[0] == y.shape[0]:
        print("Without commodity dataset was succesfully created.")
        util.make_folder(f"Data/Training/n={N}")
        np.save(f"Data/Training/n={N}/x_without", x)
        np.save(f"Data/Training/n={N}/y_without", y)

### WITH COMMODITY DATASET ###
if WITH_COMMODITY:
    ### Creating training dataset and test dataset
    util.make_folder("Data/Test")
    x, y = np.array([]), np.array([])

    for commodity, cps in data_pair.items():
        path = f"Data/{commodity}"
        util.make_folder(f"Data/Test/{commodity}/n={N}")
        commodity_df = pd.read_csv(f"Data/{commodity}/raw_data_{commodity}.csv", index_col='Date')
        for cp in cps:
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

            # Select the needed columns
            df = df[['Date', 'Adj Close', 'Volume', 'Adj Close Commodity', 'RSI', 'MFI', 'EMA', 'SO', 'MACD']]
            # Normalize them (execpt for Date)
            target_cols = ['Adj Close', 'Volume', 'Adj Close Commodity', 'RSI', 'MFI', 'EMA', 'SO', 'MACD']
            df[target_cols] = (df[target_cols]-df[target_cols].min())/(df[target_cols].max()-df[target_cols].min())

            # Take the latest 60 days as testing set and the rest for training + validation
            df_test = df.tail(60)
            df_training = df.iloc[:-60]
            # x_test, y_test = data_process.create_x_y(df_test[target_cols], N, M)
            df_test.to_csv(f"Data/Test/{commodity}/n={N}/x_without_{cp}.csv", index=True)
            x_train, y_train = data_process.create_x_y(df_training[target_cols], N, M)
            x = np.append(x, x_train)
            y = np.append(y, y_train)
    
    # Reshape
    samples = int(len(x) / (N * len(target_cols)))
    x = np.reshape(x, (samples, N, len(target_cols)))
    y = np.reshape(y, (samples, M, len(target_cols)))
    
    # Save the data
    if x.shape[0] == y.shape[0]:
        print("Without commodity dataset was succesfully created.")
        util.make_folder(f"Data/Training/n={N}")
        np.save(f"Data/Training/n={N}/x_without", x)
        np.save(f"Data/Training/n={N}/y_without", y)
