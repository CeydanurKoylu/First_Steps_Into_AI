import numpy as np

class LinearRegression:
    def __init__(self, lr: float = 0.00001, num_iter : int = 10) -> None: 
        #lr: learning rate, num_iter: the number of times the optimization algorithm will run
        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape #X's shape is (number of samples N, number of features j)
        weight = np.random.rand(num_features)
        self.weights = float(weight[0])
        print(self.weights)
        self.bias = 0.00000000000
        y_pred = np.zeros((num_features,1),dtype=float)

        for i in range(self.num_iter):
            y_pred = np.array(X * self.weights + self.bias , ndmin=2 ).T #y_pred's shape should be (N,1)
            dw = float(float(1.00 / num_samples) * float((np.dot(X.T, y_pred - y))[0][0]))
            db = float(1.00 / num_samples) * np.sum(y_pred - y)

            self.weights = float(self.weights - (self.lr)*dw)
            
            self.bias = (self.bias) - (self.lr)*float(db)
            
            #print(X.shape, X.T.shape, y.shape, y.T.shape)
            #print(dw.shape, db.shape, y_pred.shape, self.weights.shape, self.bias.shape)
            #print(dw)
            #print(y_pred.shape)
          
        return self
    
    def predict(self, X):
        return X * self.weights + self.bias