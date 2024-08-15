import unittest
import numpy as np
from algorithms import PolynomialFeatures, SGDRegressor


class TestPolynomialFeatures(unittest.TestCase):
    def test_degree_1(self):
        xtrain = np.array([[1, 2], [3, 4], [5, 6]])
        model = PolynomialFeatures(degree=1)
        transformed = model.fit_transform(xtrain)
        expected = np.array([[1, 1, 2], [1, 3, 4], [1, 5, 6]])
        np.testing.assert_almost_equal(transformed, expected)

    def test_degree_2(self):
        xtrain = np.array([[1, 2], [3, 4], [5, 6]])
        model = PolynomialFeatures(degree=2)
        transformed = model.fit_transform(xtrain)
        expected = np.array(
            [[1, 1, 2, 1, 2, 4], [1, 3, 4, 9, 12, 16], [1, 5, 6, 25, 30, 36]]
        )
        np.testing.assert_almost_equal(transformed, expected)

    def test_single_feature(self):
        xtrain = np.array([[1], [2], [3]])
        model = PolynomialFeatures(degree=2)
        transformed = model.fit_transform(xtrain)
        expected = np.array([[1, 1, 1], [1, 2, 4], [1, 3, 9]])
        np.testing.assert_almost_equal(transformed, expected)


if __name__ == "__main__":
    unittest.main()
