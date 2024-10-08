import numpy as np


class PolynomialFeatures:
    def __init__(self, degree=2):
        self.degree = degree

    def fit_transform(self, xtrain):
        n_features = xtrain.shape[1]
        dp = {}
        for d in range(self.degree + 1):
            dp[(1, d)] = np.array([[d]])
        for features in range(2, n_features + 1):
            for d in range(self.degree + 1):
                combinations = []
                for i in range(d + 1):
                    tail_combinations = dp[(features - 1, d - i)]
                    combinations.append(
                        np.hstack(
                            (
                                np.full((tail_combinations.shape[0], 1), i),
                                tail_combinations,
                            )
                        )
                    )
                combinations.reverse()
                dp[(features, d)] = np.vstack(combinations)
        self.result = np.vstack([dp[(n_features, d)] for d in range(self.degree + 1)]).T
        m, _ = xtrain.shape
        _, d = self.result.shape
        xtraintrans = np.zeros((m, d))
        for i in range(m):
            for j in range(d):
                xtraintrans[i, j] = np.prod(xtrain[i, :] ** self.result[:, j])
        return xtraintrans

    def transform(self, xtest):
        m, _ = xtest.shape
        _, d = self.result.shape
        xtesttrans = np.zeros((m, d))
        for i in range(m):
            for j in range(d):
                xtesttrans[i, j] = np.prod(xtest[i, :] ** self.result[:, j])
        return xtesttrans
