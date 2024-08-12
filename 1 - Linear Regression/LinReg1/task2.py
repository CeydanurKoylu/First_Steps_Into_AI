import numpy as np
import matplotlib.pyplot as plt

A = []

with open("second_linear_regression_data.txt", "r") as filestream:
    for line in filestream:
        currentline = line.split(",")
        A.append([float(currentline[0]),float(currentline[1])])

matrix = np.array(A,'float')
matrix = matrix[matrix[:,0].argsort()] 
X = np.array(matrix[:,0], dtype=np.float64).reshape(-1,1)
y = np.array(matrix[:,1], dtype=np.float64).reshape(-1,1)


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
        J_history[i] = (1/(2*m)) * np.sum(((X @ theta) - y)**2)
    return theta, J_history

num_samples = y.size

# Feature normalization
X = X / np.max(X)

def polynomial_kernel(X, Z, d):
    '''
    Compute dot product between each row of X and each row of Z
    '''
    m1,_ = X.shape
    m2,_ = Z.shape
    K = np.zeros((m1, m2))
    for i in range(0,m1):
        for j in range(0,m2):
            K[i,j] = (np.dot(X[i,:], Z[j,:]) + 1)**d
            
    return K
K_train = polynomial_kernel(X,X,5)
K_train = K_train + 1e-10*np.eye(num_samples)

print(K_train.shape)

features,_ = K_train.shape

K_train = K_train / np.max(K_train)

# Add the x0 term
xstack = np.hstack([np.ones_like(X), K_train])

# Initialize theta
theta = np.zeros((features + 1,1))

print(theta.shape)

# Obtain the parameter matrix
theta, J = gradient(xstack, y, theta, 9000, 0.05)


print("Costs: ",J)

plt.plot(X, y, 'bo')
plt.scatter(X, xstack @ theta)
plt.plot(X, xstack @ theta)

plt.ylabel('Y: Target')
plt.xlabel('X: Feature')
plt.legend(['Data', 'LinearFit'])
plt.title('Linear Regression')
plt.grid()
plt.show()


