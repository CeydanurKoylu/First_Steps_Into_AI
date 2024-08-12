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

# Until now we have studied Trends, Seasonality and Cycles
# If we subtract all of this from our target variable, we are left with residuals
# To get a complete model, we will do all of the components of the time series step by step, and then we will deal with residuals

# The process is like:
"""
model1.fit(X_train1, y_train)
y_pred1 = model1.predict(X_train1)

model2.fit(X_train, y_test-y_pred1)
y_pred2 = model2.predict(X_train2)

y_pred = y_pred1 + y_pred2
"""
# We will generally want to use different models and feature sets for the y_pred's to get better results, which makes the model hybrid

industries = ["BuildingMaterials", "FoodAndBeverage"]

retail_sales = pd.read_csv("us-retail-sales.csv", usecols=['Month'] + industries,
    parse_dates=['Month'], #Only uses the columns food_&_beverage and building_materials, indexes them using the months as time steps
    index_col='Month',).to_period("D").reindex(columns= industries)

retail_sales = pd.concat({"Sales": retail_sales}, names=[None, "Industries"], axis=1)


# Setting the trend feature of sales:
dp = DeterministicProcess(
        order = 2,
        drop = True,
        constant = True,
        index = retail_sales.index
)

y = retail_sales.copy()
X = dp.in_sample() # takes the data that's present in the sample

idx_train, idx_test = train_test_split(
    y.index, test_size=12 * 4, shuffle=False,
)
X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
y_train, y_test = y.loc[idx_train], y.loc[idx_test]

# Multioutput regression:
model = LinearRegression(fit_intercept= False)
model.fit(X_train,y_train)

# Now our predictions are not series anymore, they are dataframes since we are predicting two categories at the same time
y_fit = pd.DataFrame(model.predict(X_train), columns= y_train.columns, index = y_train.index)
y_fore = pd.DataFrame(model.predict(X_test), columns= y_test.columns, index = y_test.index)

# The trend in both sales, as found by our order 2 linear regression model:
"""
axs = y_train.plot(color = "0.25", subplots = True, sharex = True)
axs = y_test.plot(color = "0.25", subplots = True, sharex = True,ax =axs)
axs = y_fit.plot(color = "C0", subplots = True, sharex = True, ax = axs)
axs = y_fore.plot(color = "C3", subplots = True, sharex = True, ax = axs)
for ax in axs: ax.legend([])
_ = plt.suptitle("Trends")
plt.show()
"""

# Since XGBoot cannot do multioutput regression, we need to convert our dataset from wide format to stacked format:

X = retail_sales.stack()

# If you pop it like this, we are left with an empty dataframe, no columns, only indexes:
y = X.pop("Sales")

# The reset_index makes the indexes stand up again, and gives them a name
X = X.reset_index("Industries")

# This loop turns the object types in the indexes to 0's and 1's:
for colname in X.select_dtypes(["object", "category"]):
    X[colname], _ = X[colname].factorize()

# Will be used for annual monthly seasonalities:
X["Month"] = X.index.month
X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :] # To get all the features we need ":"
y_train, y_test = y.loc[idx_train], y.loc[idx_test]

""" If you don't get indexes for the split and just try to do it normally, you get NaN values """

# Also turn the earlier predictions on training and test datasets to long format
# Squeeze makes them convert from DataFrames to Series:
y_fit = y_fit.stack().squeeze()
y_fore = y_fore.stack().squeeze()

residuals = y_train - y_fit # The resiudals from the training data

xgb  = XGBRegressor()
xgb.fit(X_train, residuals) # BOOSTING: This training data gives these errors, so learn them and apply them with a test data too

# Newly configured target predictionsss:
y_fit_boosted = y_fit + xgb.predict(X_train)
y_fore_boosted = y_fore + xgb.predict(X_test)

# We need to unstack the target variables and predictions:
axs = y_train.unstack(['Industries']).plot(
    color='0.25', figsize=(11, 5), subplots=True, sharex=True,
    title=['BuildingMaterials', 'FoodAndBeverage'],
)
axs = y_test.unstack(['Industries']).plot(
    color='0.25', subplots=True, sharex=True, ax=axs,
)
axs = y_fit_boosted.unstack(['Industries']).plot(
    color='C0', subplots=True, sharex=True, ax=axs,
)
axs = y_fore_boosted.unstack(['Industries']).plot(
    color='C3', subplots=True, sharex=True, ax=axs,
)
for ax in axs: ax.legend([])
plt.show()

""" NOTE: There is a slight error with the Building Material forecasting in the graph, 
because of linear regression's erronous trend forecast. Hybrid models can and will continue each others mistakes, so be careful. """




