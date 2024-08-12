import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from scipy.signal import periodogram
from Ch_5_Hybrids.plotting_functions import Plots as pls

from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf

flu = pd.read_csv("flu-trends.csv")
flu.set_index(
    pd.PeriodIndex(flu.Week, freq="W"),
    inplace=True
)
flu.drop("Week", axis=1, inplace=True)

# This plot shows that flu visits are non seasonal, but cyclic:
"""
ax = flu.FluVisits.plot(title='Flu Trends', **pls.plot_params)
_ = ax.set(ylabel="Office Visits")
plt.show()
"""
# These plots indicate that using first 4 lags as features would be adequate 
# and the relationships between lags and the target variable is mostly linear:
"""
obj = pls()
_ = obj.plot_lags(flu.FluVisits, lags=12, nrows=2)
_ = plot_pacf(flu.FluVisits, lags=12)
plt.show()
"""

def make_lags(ts, lags=1):
    return pd.concat(
    {f'y_{lags}_{i}' : ts.shift(i)
     # !!!!!!!!!
    for i in range(1,lags+1) #IF YOU DON'T DO IT LIKE THIS, FIRST FEATURE WILL BE THE SAME AS Y
    }, axis = 1
    ) # returns the concatanated columns of lagged y variable

y = flu.FluVisits.copy()

X = make_lags(flu.FluVisits, lags=4)
X = X.fillna(0.0) # fills the NaN values with 0.0

print(X.shape, y.shape)

# Splitting the data into two parts with test size being 60 weeks
# no shuffle since we want to make a forecast for future values:

X_train, X_test ,y_train, y_test = train_test_split(X, y, test_size=60,shuffle=False) 
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = pd.Series(model.predict(X_train), index = y_train.index)
y_fore = pd.Series(model.predict(X_test), index = y_test.index)

# The original target variables:
"""
ax = y_train.plot(**pls.plot_params)
ax = y_test.plot(**pls.plot_params)
"""
# Our predictions for the target variable:
"""
ax = y_pred.plot()
_ = y_fore.plot(ax=ax, color = 'C3')
plt.show()
"""
# The forecasted values graph which is a bit late in rising:
"""
ax = y_test.plot(**pls.plot_params)
_ = y_fore.plot(ax=ax, color = 'C3')
plt.show()
"""
# To prevent late rising we can add some "leading indicators" that indicate a flu season is coming
# We will use flu searches on Google since they tend to rise before flu visits increase:

searches = ["FluContagious", "FluCough", "FluFever", "InfluenzaA", "TreatFlu", "IHaveTheFlu", "OverTheCounterFlu", "HowLongFlu"]
X0 = make_lags(flu[searches], lags = 3)
X0.columns = [' '.join(col).strip() for col in X0.columns.values] # This makes the columns stand further away and act like seperate features

X1 = X
X = pd.concat([X0,X1], axis = 1).fillna(0.0)

X_train, X_test ,y_train, y_test = train_test_split(X, y, test_size=60,shuffle=False) 
model.fit(X_train, y_train)
y_fore = pd.Series(model.predict(X_test), index = y_test.index)
# This is a bit of a loose fit, however it is not late in rising now:
ax = y_test.plot(**pls.plot_params)
_ = y_fore.plot(ax=ax, color = 'C3')
plt.show()

# Lookahead leakage/ bias: Using the data from future (test data) in the training model.
# Which causes a false idea of model working well, when in reality it may have a high variance when tested with other data
