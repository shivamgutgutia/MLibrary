import numpy as np

class BGDRegressor:
    def __init__(self,learning_rate=0.01, epochs = 100):
        self.coeff=None
        self.intercept = None
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self,xtrain,ytrain):
        self.intercept = 0
        self.coeff = np.ones(xtrain.shape[1])

        for _ in range(self.epochs):
            ypred = np.dot(xtrain,self.coeff)+self.intercept

            intercept_der = -2*np.mean(ytrain-ypred)
            self.intercept-= self.lr*intercept_der

            coeff_der = -2/xtrain.shape[0]*np.dot(ytrain-ypred,xtrain)
            self.coeff -= self.lr*coeff_der


    def predict(self,xtest):
        return np.dot(xtest,self.coeff)+self.intercept
