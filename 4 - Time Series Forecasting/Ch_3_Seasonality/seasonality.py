import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from scipy.signal import periodogram
from Ch_5_Hybrids.plotting_functions import MyClass as mc

tunnel = pd.read_csv("tunnel.csv", parse_dates = ["Day"])

tunnel = tunnel.set_index("Day").to_period("D")


tunnel["Day"] = tunnel.index.dayofweek
tunnel["Week"] = tunnel.index.week

tunnel["DayOfYear"] = tunnel.index.dayofyear
tunnel["Year"] = tunnel.index.year

print(tunnel.head())

#The data distributed over the year and shown for days of the week:
"""
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 6))
mc.seasonal_plot(tunnel, y="NumVehicles", period="Week", freq="Day", ax=ax0)
mc.seasonal_plot(tunnel, y="NumVehicles", period="Year", freq="DayOfYear", ax=ax1)
plt.show()
"""
#Periodgram to decide on the order of fourier series:
"""
mc.plot_periodogram(tunnel["NumVehicles"])
plt.show()
"""
# There is a stromger weekly season than the annual one
# We will model annual with Fourier Series and weekly with seasonal indicators:

fourier = CalendarFourier(freq = "A", order=10) # "A" is for the annual, we decided the order from the periodogram

dp = DeterministicProcess(
    drop=True,
    constant=True,
    order=1,
    additional_terms=[fourier], # additional_terms expects a list of elements so we need to put the brackets here
    index= tunnel.index,
    seasonal=True, # this means the process is being used for creating seasonal indicators, and since indexes are day typed
                  # days of the week (monday, tuesday etc.) is created as distinct categories and used as one hot variables
)

X = dp.in_sample()
print(X.head())

y = tunnel["NumVehicles"]

X_fore = dp.out_of_sample(steps=90)

model = LinearRegression(fit_intercept=False)

model.fit(X,y)

y_pred = pd.Series(model.predict(X), index=X.index)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax = y.plot(color='0.25', style='.', title="Tunnel Traffic - Seasonal Forecast")
ax = y_pred.plot(ax=ax, label="Seasonal")
ax = y_fore.plot(ax=ax, label="Seasonal Forecast", color='C3')
_ = ax.legend()
plt.show()

# In this example we used the Fourier Series and seasonal indicators to prepare a hybrid X model with lots of features
    # This resulted in our linear regression model to be flexible but also less biased

# If you take the difference "y - y_pred" this new entitiy is named "y_deseasoned"
    # When you take the periodograph of y_deseasoned, it appears almost flat, 
    # which means our model was successful in capturing the data seasonality
