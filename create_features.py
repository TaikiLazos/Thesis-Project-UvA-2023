import yfinance as yf
import pandas as pd

# Let's download data of crude oil and two oil companies to experiment
# start_date = "2020-01-01"
# end_date = "2023-04-01"
# crude_oil_df = yf.download("CL=F", start_date, end_date)
# oil_cp1_df = yf.download("XOM", start_date, end_date)
# oil_cp2_df = yf.download("BP", start_date, end_date)

# the size differ because they have different holidays!!!
# print(len(crude_oil_df), len(oil_cp1_df), len(oil_cp2_df))


def add_commodity(company_df, commodity_df):
    """
    It adds related commodity price of the same day to that company's dataframe.
    """
    # merge two df by Date
    merged_data = pd.merge(company_df, commodity_df["Adj Close"], on="Date")
    # rename the Adj columns
    merged_data = merged_data.rename(
        columns={"Adj Close_x": "Adj Close", "Adj Close_y": "Adj Close Commodity"}
    )
    return merged_data


# result = add_commodity(oil_cp1_df, crude_oil_df)
# print(result.columns)


def add_technical_indicators(df):
    """
    It adds technical features to the given company's dataframe.
    Five technical features are RSI, MFI, EMA, SO, MACD
    """
    # Relative Strength Index (RSI)
    # RSI = 100 - [100 / (1+RS)]
    # Where RS = Avg. of x days’ up closes / Average of x days' down closes.
    # I chose x to be two weeks, so x = 14.
    tmp_df = df.copy()
    tmp_df["delta"] = tmp_df["Adj Close"].diff()
    tmp_df["gain"] = tmp_df["delta"].apply(lambda x: x if x > 0 else 0)
    tmp_df["loss"] = tmp_df["delta"].apply(lambda x: abs(x) if x < 0 else 0)
    x = 14  # Number of days for the average
    tmp_df["avg_gain"] = tmp_df["gain"].rolling(window=x).mean()
    tmp_df["avg_loss"] = tmp_df["loss"].rolling(window=x).mean()
    tmp_df["RS"] = tmp_df["avg_gain"] / tmp_df["avg_loss"]
    df["RSI"] = 100 - (100 / (1 + tmp_df["RS"]))

    # Money Flow Index (MFI)
    # Money Flow (MF) = Typical Price * Volume | Typical price = (High + Low + Adj Close) / 3
    # Money Ratio (MR) = (Positive MF / Negative MF)
    # MFI = 100 – (100/ (1+MR))
    tmp_df["typical_price"] = (tmp_df["High"] + tmp_df["Low"] + tmp_df["Adj Close"]) / 3
    tmp_df["raw_money_flow"] = tmp_df["typical_price"] * tmp_df["Volume"]
    tmp_df["pmf"] = (
        tmp_df["raw_money_flow"]
        .where(tmp_df["typical_price"] > tmp_df["typical_price"].shift(1), 0)
        .rolling(window=x)
        .sum()
    )
    tmp_df["nmf"] = (
        tmp_df["raw_money_flow"]
        .where(tmp_df["typical_price"] < tmp_df["typical_price"].shift(1), 0)
        .rolling(window=x)
        .sum()
    )
    tmp_df["mfr"] = tmp_df["pmf"] / tmp_df["nmf"]
    df["MFI"] = 100 - (100 / (1 + tmp_df["mfr"]))

    # Exponential Moving Average (EMA)
    # 1. Compute the SMA. Avg.Closing of period x -> not needed to calculate EMA
    # 2. Calculate the multiplier for weighting the EMA. α = 2 / [x + 1]
    # 3. Find EMA with EMA = Price(today)*k + EMA(yesterdat)*(1−alpha)
    # tmp_df['SMA'] = tmp_df['Adj Close'].rolling(window=x).mean()
    alpha = 2 / (x + 1)
    df["EMA"] = tmp_df["Adj Close"].ewm(alpha=alpha, adjust=False).mean()

    # Stochastic Oscillator (SO)
    # SO = [(Close - Lowest of Low of period x) / (Highest of High of period x - Lowest of Low of period x)] * 100
    tmp_df["highest_high"] = tmp_df["High"].rolling(window=x).max()
    tmp_df["lowest_low"] = tmp_df["Low"].rolling(window=x).min()
    df["SO"] = (
        (tmp_df["Adj Close"] - tmp_df["lowest_low"])
        / (tmp_df["highest_high"] - tmp_df["lowest_low"])
        * 100
    )

    # Moving Average Convergence/Divergence (MACD)
    # Positive = bullish, Negative = Bearish
    # MACD = short-Period EMA − long-Period EMA
    # I chose short = 12 and long = 26 (https://www.investopedia.com/terms/m/macd.asp)
    short_period = 12
    tmp_df["EMA_short"] = (
        tmp_df["Adj Close"].ewm(span=short_period, adjust=False).mean()
    )
    long_period = 26
    tmp_df["EMA_long"] = tmp_df["Adj Close"].ewm(span=long_period, adjust=False).mean()
    df["MACD"] = tmp_df["EMA_short"] - tmp_df["EMA_long"]

    # Drop all rows contianing NaN
    df = df.dropna()
    return df


# result = add_technical_indicators(oil_cp1_df)
# print(result)
