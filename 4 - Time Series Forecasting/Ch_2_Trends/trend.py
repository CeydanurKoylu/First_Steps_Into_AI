import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess
set_matplotlib_formats('retina')

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

tunnel = pd.read_csv("tunnel.csv", parse_dates = ["Day"])

tunnel = tunnel.set_index("Day").to_period()
print(tunnel.head())

moving_average = tunnel.rolling(
    window= 365, # the moving window size
    center = True, # the values is always recorded at the center of the window
    min_periods=183, # min # of observations in a window for it t o not give  a nan value
).mean()

ax = tunnel.plot(style=".", color="0.5")
moving_average.plot(ax=ax,  linewidth=3, title="Tunnel Traffic - 365-Day Moving Average", legend=False)
plt.show()

# we will use this function to engineer the data to get X, from now on:

dp = DeterministicProcess(
    index = tunnel.index, # gets the dates from the training data
    constant = True, # the dummy feaature for the intercept/ bias
    order = 1, # the time dummy, since the trend seems to be in a linear shape
    drop = True, # drop terms if necessary to avoid colinearity
) 

X = dp.in_sample() # creates features for the dates given in the index argument

# the deterministic process is just another technical name for time series that behave non-randomly and deterministic
# features that are accessed through time index will generally be deterministic

print(X.head())

y = tunnel["NumVehicles"]
y,X = y.align(X,join="inner")

model = LinearRegression(fit_intercept=False) # since we have already fit hte intercept in X using DP, 
                                              # duplicating the value would only bring trouble

model.fit(X,y)
y_pred = pd.Series(model.predict(X), index=X.index)

ax = tunnel.plot(style=".", color="0.5", title="Tunnel Traffic - Linear Trend")
_ = y_pred.plot(ax=ax, linewidth=3, label="Trend")
plt.show()

#The linear regression model found almost the same trend as our moving averages function, which means linear features was a good idea
 
X = dp.out_of_sample(steps=30) # gives dates that are not in the original sample
y_fore = pd.Series(model.predict(X), index=X.index) # forecasts num_vehicles for the upcoming dates, using linear regression
print(y_fore.head())

ax = tunnel["2005-05":].plot(title="Tunnel Traffic - Linear Trend Forecast", **plot_params)
ax = y_pred["2005-05":].plot(ax=ax, linewidth=3, label="Trend")
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color="C3")
_ = ax.legend()
plt.show()

"""TTo get better results we can try to increase the degree of the polynomial that we use"""
