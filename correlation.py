import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np


# data start from, ends at
start_date = '2000-08-23'  # This is the oldest date yfinance has
end_date = '2023-04-01'

# We use crude oil and oil company dataset 
data_pair = {'CL=F': ['2222.SR', 'SNPMF', 'PCCYF', 'XOM', 'SHEL', 'TTE', 'CVX', 'BP', 'MPC', 'VLO']}

# Loading commodity data
commodity_data = pd.DataFrame()
oil = yf.download('CL=F', start_date, end_date)['Adj Close']
oil = (oil-oil.min())/(oil.max()-oil.min())

commodity_data = pd.concat([commodity_data, oil])
commodity_data = commodity_data.to_numpy()

# empty df for companies
company_data = pd.DataFrame()

for cp in data_pair['CL=F']:
    # Download the data
    tmp_df = yf.download(cp, start_date, end_date)['Adj Close']
    # Normalization
    tmp_df = (tmp_df-tmp_df.min())/(tmp_df.max()-tmp_df.min())
    company_data = pd.concat([company_data, tmp_df])

company_data = company_data.groupby(company_data.index).mean()

company_data = company_data.to_numpy()

# # plotting the arrays
# start = abs(len(commodity_data) - len(company_data))
# plt.plot(commodity_data, label='Oil')
# plt.plot(company_data, label='Stock Avg')

# plt.title('Average oil companies stock price plotted against oil price')
# plt.legend()
# plt.show()

# Pearson Correlation
# Calculate Pearson correlation coefficient and p-value


start = abs(len(commodity_data) - len(company_data))

commodity_data = np.reshape(commodity_data, (len(commodity_data),))
company_data = np.reshape(company_data, (len(company_data),))

correlation, p_value = pearsonr(commodity_data, company_data[start:])

print("Pearson correlation coefficient:", correlation)
print("p-value:", p_value)

correlation, p_value = spearmanr(commodity_data, company_data[start:])
print("Spearman's rank correlation coefficient:", correlation)
print("p-value:", p_value)