import numpy as np


class LogisticRegression:
    def __init__(self, epochs=1000, learning_rate=0.01):
        self.coeff = None
        self.intercept = None
        self.epochs = epochs
        self.lr = learning_rate

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, xtrain, ytrain):
        xtrain = np.insert(xtrain, 0, 1, axis=1)
        weights = np.ones(xtrain.shape[1])
        m = xtrain.shape[0]
        for _ in range(self.epochs):
            ybar = self.sigmoid(np.dot(xtrain, weights))
            weights += self.lr / m * np.dot(ytrain - ybar, xtrain)
        self.intercept = weights[0]
        self.coeff = weights[1:]

    def predict(self, xtest):
        xtest = np.insert(xtest, 0, 1, axis=1)
        return np.where(np.dot(xtest, np.append(self.intercept, self.coeff)) > 0, 1, 0)
