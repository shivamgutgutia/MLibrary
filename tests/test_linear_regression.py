import unittest
from algorithms import LinearRegression
import numpy as np

class TestLinearRegression(unittest.TestCase):

    def test_simple_regression(self):
        X = np.array([[1],[2],[3],[4]])
        y = np.array([3,6,9,12])

        reg = LinearRegression()
        reg.fit(X,y)
        ypred = reg.predict(X)

        for pred,actual in zip(ypred,y):
            self.assertAlmostEqual(pred,actual)
    

    def test_multiple_regression(self):
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3

        reg = LinearRegression()
        reg.fit(X, y)
        ypred = reg.predict(X)

        for pred,actual in zip(ypred,y):
            self.assertAlmostEqual(pred,actual)

if __name__ == "__main__":
    unittest.main()
