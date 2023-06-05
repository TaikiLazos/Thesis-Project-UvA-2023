import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get the raw df
commodity = "CL=F"
cp = "SHEL"
path = f"Data/{commodity}/raw_data_{cp}.csv"
raw_df = pd.read_csv(path)
raw_df['Adj Close'] = (raw_df['Adj Close']-raw_df['Adj Close'].min())/(raw_df['Adj Close'].max()-raw_df['Adj Close'].min())

# Let's take data from this year (2023)
raw_df = raw_df[raw_df['Date'] >= "2022-01-01"]['Adj Close'].to_numpy()


# let's load the result of our model
ver = ['without', 'with']
N = [5, 7, 10]


# Add this 
df = pd.DataFrame()

column_names = [cp]
for v in ver:
    for n in N:
        column_names.append(f"{v}_{n}")


data = {}
data['y'] = raw_df[::-1]
for v in ver:
    for n in N:
        y_hat = np.load(f"Data/Results/{commodity}/{v}_n={n}.npy")
        data[f'y_hat_{n}_{v}'] =y_hat[::-1]

for k in data.keys():
    df = pd.concat([df, pd.DataFrame(data[k]).T], ignore_index=True).fillna(np.NaN)


real = df.T.iloc[::-1].reset_index(drop=True)
real = real.set_axis(column_names, axis=1)
print(real)
real.plot()
plt.savefig(f'result.png')
