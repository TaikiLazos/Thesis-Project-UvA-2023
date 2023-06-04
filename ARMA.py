from statsmodels.tsa.arima.model import ARIMA
import numpy as np

n = 5
ver = "without" # with -> with commodity feature
# the data is normalized, and has everything except the latest 20% SHEL data
x = np.load(f"Data/Training/n={n}/x_{ver}.npy")
y = np.load(f"Data/Training/n={n}/y_{ver}.npy")

# 1,1,2 ARIMA Model
model = ARIMA(x[:, 0], order=(1,1,2))
model_fit = model.fit()
print(model_fit.summary())