import numpy as np
import matplotlib.pyplot as plt

XX = []
yy = []

with open("first_linear_regression_data.txt", "r") as filestream:
    for line in filestream:
        currentline = line.split(",")
        XX.append(float(currentline[0]))
        yy.append(float(currentline[1]))

X = np.array(XX, dtype=np.float64).reshape(-1,1)
y = np.array(yy, dtype=np.float64).reshape(-1,1)

def costFunction(X, y, theta):
    m = y.size
    return (1/(2*m)) * np.sum(((X @ theta) - y)**2)

def gradient(X, y, theta, iterations, alpha):
    J_history = np.zeros(iterations)
    m = y.size
    for i in range(iterations):
        error = (X @ theta) - y
        theta -= (alpha/m) * (X.T @ error)
        """
        THIS WORKS BUT IT'S REALLY SLOW SINCE IT DOESN'T UTILIZE NP
        te0, te1 = 0, 0 
        for j in range(0,m):
            te0 +=  (error[j] * X[j][0])
            te1 +=  (error[j] * X[j][1])
        temp0 = theta[0] - ((alpha/m) * te0)
        temp1 = theta[1] - ((alpha/m) * te1)
        theta = np.array([temp0,temp1]).reshape(2,1)
        """
        J_history[i] = costFunction(X, y, theta)
    return theta, J_history

num_samples = y.size

# Feature normalization
X = X / np.max(X)

# Add the x0 term
xstack = np.hstack([np.ones_like(X), X])

# Initialize theta
theta = np.zeros((2,1))


# Obtain the parameter matrix
theta, J = gradient(xstack, y, theta, 9000, 0.05)
print("Costs: ",J)
# Plotting

plt.plot(X, y, 'bo')
plt.plot(X, xstack @ theta, '-')
plt.ylabel('Y: Target')
plt.xlabel('X: Feature')
plt.legend(['Data', 'LinearFit'])
plt.title('Linear Regression')
plt.grid()
plt.show()


