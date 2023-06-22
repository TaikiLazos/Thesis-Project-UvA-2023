import numpy as np
import matplotlib.pyplot as plt
import util
import pandas as pd


def plot_results(commodity, company, N, M, VER):
    """
    It plots three lines in one plot to show the result
    """
    util.make_folder(f"Results/Plots/n={N}&m={M}")
    oil = pd.read_csv(f"Data/{commodity}/raw_data_{commodity}.csv")
    y_true = np.load(f"Results/{commodity}/n={N}&m={M}/y_true_{VER}_{company}.npy")
    y_predict = np.load(
        f"Results/{commodity}/n={N}&m={M}/y_predict_{VER}_{company}.npy"
    )
    y_forecast = np.load(
        f"Results/{commodity}/n={N}&m={M}/y_forecast_{VER}_{company}.npy"
    )
    oil["Adj Close"] = (oil["Adj Close"] - oil["Adj Close"].min()) / (
        oil["Adj Close"].max() - oil["Adj Close"].min()
    )
    oil = oil["Adj Close"].iloc[-y_true.shape[0] :]
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
        f"Results/Plots/n={N}&m={M}/result of {company}, N = {N}, M = {M}, VER = {VER}.png"
    )
    plt.clf()


if __name__ == "__main__":
    company = [
        "2222.SR",
        "XOM",
        "SHEL",
        "TTE",
        "CVX",
        "BP",
        "MPC",
        "VLO",
    ]
    N = 1
    M = 2
    VER = ["with" ,"without"]
    for x in range(1, 6):
        for v in VER:
            for cp in company:
                plot_results("CL=F", cp, N*x, M*x, v)

# # Get the raw df
# commodity = "CL=F"
# cp = "SHEL"
# path = f"Data/{commodity}/raw_data_{cp}.csv"
# raw_df = pd.read_csv(path)
# raw_df['Adj Close'] = (raw_df['Adj Close']-raw_df['Adj Close'].min())/(raw_df['Adj Close'].max()-raw_df['Adj Close'].min())

# N = 19
# ver = "without"
# y_true = np.load(f"Results/{commodity}/N={N}/y_true_{ver}_{cp}.npy")
# y_fore = np.load(f"Results/{commodity}/N={N}/y_forecast_{ver}_{cp}.npy")
# y_pred = np.load(f"Results/{commodity}/N={N}/y_predict_{ver}_{cp}.npy")


# days = np.array(np.arange(len(y_true)))
# plt.plot(days,y_true, label='true')
# plt.plot(days, y_pred, label='prediction')
# plt.plot(days, y_fore, label='forecasting')
# plt.legend()
# plt.savefig(f'Results/result of {cp}, N = {N} {ver}.png')

# print(1 + '1')

# # Let's take data from this year (2023)
# raw_df = raw_df[raw_df['Date'] >= "2022-02-01"]['Adj Close'].to_numpy()


# # let's load the result of our model
# ver = ['without', 'with']
# N = [5, 7, 10]


# # Add this
# df = pd.DataFrame()

# column_names = [cp]
# for v in ver:
#     for n in N:
#         column_names.append(f"{v}_{n}")


# data = {}
# data['y'] = raw_df[::-1]
# for v in ver:
#     for n in N:
#         y_hat = np.load(f"Data/Results/{commodity}/{v}_n={n}.npy")
#         data[f'y_hat_{n}_{v}'] =y_hat[::-1]

# for k in data.keys():
#     df = pd.concat([df, pd.DataFrame(data[k]).T], ignore_index=True).fillna(np.NaN)


# real = df.T.iloc[::-1].reset_index(drop=True)
# real = real.set_axis(column_names, axis=1)
# print(real)
# real.plot()
# plt.savefig(f'result.png')


# days = np.array(np.arange(len(y_true)))
# plt.plot(days,y_true, label='true')
# plt.plot(days, prediction, label='prediction')
# plt.plot(days, forecast, label='forecasting')
# plt.legend()
# plt.savefig(f'Results/result of {company}, N = {N}, {VER}.png')
