import yfinance as yf
import os
import matplotlib.pyplot as plt
import pandas as pd

# data start from, ends at
start_date = '2000-08-23'  # This is the oldest datet
end_date = '2023-04-01'

# Get the data for the crude oil & Exxon Mobil
oil_data = yf.download('CL=F', start_date, end_date)
exxon_data = yf.download('XOM', start_date, end_date)

# We take the close price of these data
oil_data_processed = oil_data['Adj Close'].round(2)
exxon_data_processed = exxon_data['Adj Close'].round(3)

plt.plot(exxon_data_processed, )

newpath = os.getcwd() + '/Data'
if not os.path.exists(newpath):
    os.makedirs(newpath)

# write the data into that file
oil_data_processed.to_csv(newpath + '/oil_data.csv')
exxon_data_processed.to_csv(newpath + '/exxon_data.csv')
