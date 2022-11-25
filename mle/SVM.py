import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt
%matplotlib inline



class SVM:
    def __init__(self, max_iter=100, kernel="linear") -> None:
        self.max_iter = max_iter
        self._kernel = kernel

        # args
    def init_args(self, X, Y):
        N, M = X.shape
        self.A= np.random.rand(N)
        self.W = np.random.rand(M)
        self.b = 0.0
        self.C = 1.0

        self.X, self.Y = X, Y

    def _KKT(self, i):
        pass

    def predict(self, x):
        """
        use rule to determine the class of input
            \sum_i \alpha_i y_i x_i ⋅ x + b > 0 -> positive sample
            \sum_i \alpha_i y_i x_i ⋅ x + b ≤ 0 -> negative sample
        Parameters:
        -----------
        x: (N, M)
            N: number of samples
            M: dimension of features
        Returns:
        --------
        output: np.float
            output > 0 : input is a positive sample
            output < 0 : input is a negative sample
        """
        return self.A * self.Y * self.kernel(x, self.X) + self.b


    def kernel(self, Z1, Z2, ker = "linear"):
        """
        Compute kernel matrix w.r.t to specified kernel
        Parameters:
        -----------
        Z1, Z2: (N1, M), (N2, M)
            N: number of samples
            M: dimension of features
        Returns:
        --------
        kernel matrix K: (N1, N2)
            K[i][j] is the inner product of sample i and j
        """
        if ker == "linear":
            return Z1 @ Z2.T
        else:
            raise NotImplementedError