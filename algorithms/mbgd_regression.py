import numpy as np
import random
class MBGDRegressor:
    def __init__(self,batch_size,learning_rate=0.01, epochs = 100):
        self.coeff=None
        self.intercept = None
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self,xtrain,ytrain):
        self.intercept = 0
        self.coeff = np.ones(xtrain.shape[1])
        if self.batch_size>xtrain.shape[0]:
            return

        for _ in range(self.epochs):
            for _ in range(xtrain.shape[0]//self.batch_size):
                idx = random.sample(range(xtrain.shape[0]),self.batch_size)
                ypred = np.dot(xtrain[idx],self.coeff)+self.intercept

                intercept_der = -2*np.mean(ytrain[idx]-ypred)
                self.intercept-= self.lr*intercept_der

                coeff_der = -2/self.batch_size*np.dot(ytrain[idx]-ypred,xtrain[idx])
                self.coeff -= self.lr*coeff_der


    def predict(self,xtest):
        return np.dot(xtest,self.coeff)+self.intercept
