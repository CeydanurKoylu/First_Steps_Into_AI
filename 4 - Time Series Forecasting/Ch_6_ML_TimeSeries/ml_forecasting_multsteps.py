import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from scipy.signal import periodogram
from plotting_functions import Plots as pls

from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Creating a feature + target data set that can be used to make forecasts 3 steps into the future:
"""
N = 20
ts = pd.Series(
    np.arange(N),
    dtype = pd.Int8Dtype,
    index = pd.period_range(freq = "A", start ="2010", name = "Year", periods = N)
)

X = pd.DataFrame({
    "y_lag_1" : ts.shift(1),
    "y_lag_2" : ts.shift(2),
    "y_lag_3" : ts.shift(3),
    "y_lag_4" : ts.shift(4),
    "y_lag_5" : ts.shift(5),
    "y_lag_6" : ts.shift(6)
})
y = pd.DataFrame({
    "y_lead_3" : ts.shift(-2),
    "y_lead_2" : ts.shift(-1),
    "y_lead_1" : ts,
    
})

data = pd.concat({"Target": y, "Features" : X}, axis  = 1)
print(data.head())
"""

flu_trends = pd.read_csv("./flu-trends.csv")

flu_trends.set_index(
    pd.PeriodIndex(flu_trends.Week, freq="W"),
    inplace=True
)

flu_trends.drop("Week", axis=1, inplace=True)
print(flu_trends.head())

def make_lags(y, lag, lead_time = 1):
    return pd.concat(
        {
            f"y_lag_{i}" : y.shift(i)
            for i in range(lead_time, lead_time + lag)
        }, axis=1
    )


y = flu_trends.FluVisits.copy()
X = make_lags(y, 4).fillna(0.0) # We want to make forecasts by looking to data 4 weeks prior

def make_multistep_target(y, step):
    return pd.concat(
        {
            f"y_lead_{i+1}" : y.shift(-i)
            for i in range(step)
        }, axis=1
    )
# The target data is 0-7 weeks into the future:
y = make_multistep_target(y , step=8).dropna()

y,X = y.align(X, join="inner", axis=0) # Accepts only the index values that are present in both X and y
#print(X.head(),y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_fit = pd.DataFrame(model.predict(X_train), index = X_train.index, columns=y_train.columns)
y_pred = pd.DataFrame(model.predict(X_test), index = X_test.index, columns=y_test.columns)

train_mse = mean_squared_error(y_fit, y_train, squared=False)
test_mse = mean_squared_error(y_pred, y_test, squared=False)

# The MSE's are almost 300-500 in 3000, which is kind of a lot
"""
print(train_mse, test_mse)"""

# Very bad at predicting the rises in flu visits, it always lags
# However, it appears to be good enough at predicting the decreases in flu visits:
"""
palette = dict(palette='husl', n_colors=64)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6))
ax1 = flu_trends.FluVisits[y_fit.index].plot(**pls.plot_params, ax=ax1)
ax1 = pls.plot_multistep(y_fit, ax=ax1, palette_kwargs=palette)
_ = ax1.legend(['FluVisits (train)', 'Forecast'])
ax2 = flu_trends.FluVisits[y_pred.index].plot(**pls.plot_params, ax=ax2)
ax2 = pls.plot_multistep(y_pred, ax=ax2, palette_kwargs=palette)
_ = ax2.legend(['FluVisits (test)', 'Forecast'])
plt.show()"""

from sklearn.multioutput import MultiOutputRegressor

# Direct Strategy: It trains a seperate model for each of the wanted target values
# It trains a model for one step forecast, another for two step forecast... and so on:

model = MultiOutputRegressor(XGBRegressor()) # Since XGBRegressor cannot output multiple target values
model.fit(X_train, y_train)

y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=y.columns)
y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)

# This model overfits like crazy, however it still does better on the test data compared to LinReg
"""
train_rmse = mean_squared_error(y_train, y_fit, squared=False)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
print((f"Train RMSE: {train_rmse:.2f}\n" f"Test RMSE: {test_rmse:.2f}"))

palette = dict(palette='husl', n_colors=64)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6))
ax1 = flu_trends.FluVisits[y_fit.index].plot(**pls.plot_params, ax=ax1)
ax1 = pls.plot_multistep(y_fit, ax=ax1, palette_kwargs=palette)
_ = ax1.legend(['FluVisits (train)', 'Forecast'])
ax2 = flu_trends.FluVisits[y_pred.index].plot(**pls.plot_params, ax=ax2)
ax2 = pls.plot_multistep(y_pred, ax=ax2, palette_kwargs=palette)
_ = ax2.legend(['FluVisits (test)', 'Forecast'])
plt.show()"""

# Recursive Strategy: Feeding the same model with the predictions + (old features - 1) of the model trained on the old features
# DirRec Strategy: Training seperate models for each step as in direct strategy but also feeding each new 
#                  model with the predictions of the older ones. (Kind of like a chain)

