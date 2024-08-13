import unittest
import numpy as np
from algorithms import SGDRegressor

class TestBGDRegressor(unittest.TestCase):
    def test_simple_regression(self):
        xtrain = np.array([[1],[2],[3],[4],[5]])
        ytrain = np.array([5,7,9,11,13])
        model = SGDRegressor(learning_rate=0.01,epochs=500)
        model.fit(xtrain,ytrain)
        ypred = model.predict(xtrain)
        self.assertAlmostEqual(model.coeff[0],2,places=1)
        self.assertAlmostEqual(model.intercept,3,places=1)
        for pred,actual in zip(ypred,ytrain):
            self.assertAlmostEqual(pred,actual,places=1)

    def test_multiple_regression(self):
        xtrain = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        ytrain = np.array([3, 5, 7, 9, 11])
        model = SGDRegressor(learning_rate=0.01, epochs=500)
        model.fit(xtrain,ytrain)
        ypred = model.predict(xtrain)
        for coeffpred,coeffactual in zip(model.coeff,[1,1]):
            self.assertAlmostEqual(coeffpred,coeffactual,places=1)
        self.assertAlmostEqual(model.intercept,0,places=1)
        for pred,actual in zip(ypred,ytrain):
            self.assertAlmostEqual(pred,actual,places=1)

if __name__ == '__main__':
    unittest.main()