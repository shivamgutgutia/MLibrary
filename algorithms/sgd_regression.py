import numpy as np


class SGDRegressor:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.coeff = None
        self.intercept = None
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, xtrain, ytrain):
        self.intercept = 0
        self.coeff = np.ones(xtrain.shape[1])
        for _ in range(self.epochs):
            for i in range(xtrain.shape[0]):
                idx = np.random.randint(0, xtrain.shape[0])
                ypred = np.dot(xtrain[idx], self.coeff) + self.intercept
                intercept_der = -2 * (ytrain[idx] - ypred)
                self.intercept -= self.lr * intercept_der
                coeff_der = -2 * np.dot((ytrain[idx] - ypred), xtrain[idx])
                self.coeff -= self.lr * coeff_der

    def predict(self, xtest):
        return np.dot(xtest, self.coeff) + self.intercept
