import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

XX = []
yy = []

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')



with open("first_linear_regression_data.txt", "r") as filestream:

    for line in filestream:
        currentline = line.split(",")
        XX.append(float(currentline[0]))
        yy.append(float(currentline[1]))
X = np.array(XX, dtype=np.float64, ndmin = 2).T
y = np.array(yy, dtype=np.float64, ndmin = 2).T
#print(X, y)

plt.grid()
plt.title('Linear Regression Model')
plt.xlabel('X')
plt.ylabel('Y')


plt.plot(X.T, y.T, 'o')
plt.show()

print(X.T, y.T)


plt.grid()
plt.title('Linear Regression Model')
plt.xlabel('X')
plt.ylabel('Y')

regressor = LinearRegression(num_iter=1000).fit(X, y)

plt.grid()
plt.title('Linear Regression Model')
plt.xlabel('X')
plt.ylabel('Y')

plt.scatter(X, y)
plt.plot(X, regressor.predict(X), linewidth=2,
            color='black', label='prediction')
plt.legend()
# plt.grid()
plt.show()
