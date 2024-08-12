import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib_inline
from IPython.display import set_matplotlib_formats
from sklearn.linear_model import LinearRegression
matplotlib_inline.backend_inline.set_matplotlib_formats()


# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
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


tunnel = pd.read_csv("./tunnel.csv",
                 parse_dates = ["Day"])
tunnel = tunnel.set_index("Day")
tunnel = tunnel.to_period()

tunnel["Time"] = np.arange(len(tunnel.index))

X = tunnel.loc[:, ["Time"]] # the model expects a two dimensional array since columns must be features an rows must be samples, so we put the extra brackets
y = tunnel.loc[:, "NumVehicles"] # the model expects a Series, a single column of one dimension is enough so we don't put brackets

model = LinearRegression()
model.fit(X,y)

y_pred = pd.Series(model.predict(X), index = X.index)

print(model.coef_)
print(model.intercept_)

ax = y.plot(**plot_params)
ax = y_pred.plot(ax = ax, linewidth = 3)
ax.set_title("The Time Step Graph")
plt.show()

# Since y is a series with a period index, the x-axis of the graph is automatically determined

tunnel["Lag_1"] = tunnel["NumVehicles"].shift(1) 

X1 =  tunnel.loc[:, ["Lag_1"]]
X1.dropna(inplace=True) # drops any rows of X1 which has NaN values, does this operation on the original X1 not on a copy.
y1 = tunnel.loc[:, "NumVehicles"]
y1, X1 = y1.align(X1, join="inner") # "join= 'inner'" part means that both X1 and y1 can only have row indexes that both of them already have


model1 = LinearRegression()
model1.fit(X1,y1)
y_pred1 = pd.Series(model1.predict(X1), index = X1.index)

fig, ax = plt.subplots()
ax.plot(X1['Lag_1'], y1, '.', color='0.25') #scatters the actual vehicle_s wrt the lagging ones
ax.plot(X1['Lag_1'], y_pred1)
ax.set_aspect('equal')
ax.set_ylabel('NumVehicles')
ax.set_xlabel('Lag_1')
ax.set_title('Lag Plot of Tunnel Traffic')
plt.show()

# Training a linear regression model by lagging time series gives us a linear line wrt the lag factor
# However if we plot our predictions wrt time:

ax = y1.plot(**plot_params)
ax = y_pred1.plot()
plt.show()
