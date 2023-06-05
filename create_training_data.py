import yfinance as yf
import os
import data_process
import numpy as np

# Global variables
get_RAW = False
n = 14 # window size: 5, 7, 10, 14
m = 3 # prediction window size
create_Data = True
create_Data_Combined = True

# data start from, ends at
start_date = '2000-08-23'  # This is the oldest date yfinance has
end_date = '2023-04-01'

# We use crude oil and oil company dataset 
data_pair = {'CL=F': ['2222.SR', 'SNPMF', 'PCCYF', 'XOM', 'SHEL', 'TTE', 'CVX', 'BP', 'MPC', 'VLO']}
            #  'NG=F': ['RELIANCE.NS', 'VLO', 'PSX', 'MPC', 'BP', 'CVX', 'TTE'],
            #  'GC=F': ['NEM', 'GOLD', 'AEM', 'AU', 'PLZL.IL', 'GFI', 'KGC', 'NCM', 'FCX'],
            #  'SI=F': ['HL', 'CDE', 'SVL.AX', '0815.HK', 'NILI.V', 'WPM.TO'],
            #  'KC=F': ['SBUX']}

### SAVING RAW DATA INTO LOCAL FOLDER ###
if get_RAW:
    # writing raw data into each folder
    for commodity, cps in data_pair.items():
        # make a new folder if not exists
        if not os.path.exists(f"Data/{commodity}"):
            os.makedirs(f"Data/{commodity}")
        # make a commodity dataset
        raw_data = yf.download(commodity, start_date, end_date)
        raw_data.to_csv(f"Data/{commodity}/raw_data_{commodity}.csv", index=True)
        # do that for companies
        for cp in cps:
            raw_data = yf.download(cp, start_date, end_date)
            raw_data.to_csv(f"Data/{commodity}/raw_data_{cp}.csv", index=True)

### WITHOUT COMMODITY DATASET ###
if create_Data:
    ### Creating training dataset and test dataset
    x, y = np.array([]), np.array([])

    for commodity, cps in data_pair.items():
        path = f"Data/{commodity}"
        for cp in cps:
            if os.path.exists(f"{path}/raw_data_{cp}.csv"):
                cp_x, cp_y = data_process.load_x_y(f"{path}/raw_data_{cp}.csv", n, m, True)

                # The last 30 days will be our test set the rest will be used as training + validation
                x = np.append(x, np.array([np.array(xi) for xi in cp_x])[:-30])
                y = np.append(y, np.array([np.array(yi) for yi in cp_y])[:-30])

                test_x = np.array(cp_x[-30:])
                test_y = np.array(cp_y[-30:])

                # save
                np.save(f"Data/Test/n={n}/x_without_{cp}", test_x)
                np.save(f"Data/Test/n={n}/y_without_{cp}", test_y)


    # save x and y as npy file
    v = int(x.shape[0] / n)
    x = x.reshape((v, n))
    y = y.reshape((v, m))
    print(x.shape, y.shape)

    np.save(f"Data/Training/n={n}/x_without", x)
    np.save(f"Data/Training/n={n}/y_without", y)

### WITH COMMODITY DATASET ###
if create_Data_Combined:
    ### Creating training dataset and test dataset
    x, y = np.array([]), np.array([])

    for commodity, cps in data_pair.items():
        commodity_path = f"Data/{commodity}/raw_data_{commodity}.csv"
        for cp in cps:
            company_path = f"Data/{commodity}/raw_data_{cp}.csv"
            cp_x, cp_y = data_process.load_x_y_combined(company_path, commodity_path, n, m, True)
            cp_x, cp_y = np.array(cp_x), np.array(cp_y)

            # 80% -> training + validation
            x = np.append(x, np.array([np.array(xi) for xi in cp_x])[:-30])
            y = np.append(y, cp_y[:-30])

            # 20% -> test
            test_x = cp_x[-30:]
            test_y = cp_y[-30:]
            np.save(f"Data/Test/n={n}/x_with_{cp}", test_x)
            np.save(f"Data/Test/n={n}/y_with_{cp}", test_y)
    
    # save x and y as npy file
    v = int(x.shape[0] / (2 * n))
    x = x.reshape((v, 2, n))
    y = y.reshape((v, 2, m))
    print(x.shape, y.shape)
    np.save(f"Data/Training/n={n}/x_with", x)
    np.save(f"Data/Training/n={n}/y_with", y)
