import numpy as np

class LinearRegression:
    def __init__(self):
        self.coeff=None
        self.intercept=None

    def fit(self,xtrain,ytrain):
        xtrain = np.insert(xtrain,0,1,axis=1)
        betas = np.linalg.inv(np.dot(xtrain.T,xtrain)).dot(xtrain.T).dot(ytrain)
        self.coeff = betas[1:]
        self.intercept = betas[0]

    def predict(self,xtest):
        return np.dot(xtest,self.coeff.T)+self.intercept
    
