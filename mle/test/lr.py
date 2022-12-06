

import numpy as np
import pandas as pd

def euclid_dist(X1, X2):
    """
    
    Parameters:
    -----------
    X1, X2 : (N, M)

    Returns:
    --------
    
    """

    return np.linalg.norm(X1-X2, ord=2, axis = 1)

class LinerRegression:

    def __init__(self, fit_intercept):

        self.fit_intercept = fit_intercept
        self.beta = None
        self.sigma_inv = None
        self._is_fit = False

    def fit(self, X, Y):

        N, M = X.shape[0], X.shape[1]
        # X, Y = np.atleast_2d(X), np.atleast_2d(Y)

        if self.fit_intercept:
            X = np.c_(np.ones(N), X) # -> X = np.c_[np.ones(N), X]
        
        self.sigma_inv = np.linalg.pinv(X.T @ X)
        self.beta = self.sigma_inv @ X.T @ Y
        self._is_fit = True


    def predict(self, X):

        return X @ self.beta


    def update(self, X, Y):
        # X, Y = np.atleast_2d(X), np.atleast_2d(Y)
        N, M = X.shape[0], X.shape[1]
        if self.fit_intercept:
            X = np.c_(np.ones(N), X)

        beta = self.beta
        S_inv = self.sigma_inv
        I = np.eyes(N) #  I = np.eye(X.shape[0])

        S_inv -= S_inv @ X.T @ np.linalg.pinv(I + X @ X.T) @ X @ S_inv   # np.linalg.pinv(I + X @ S_inv @ X.T)
        Y_pred = X @ beta
        beta += S_inv @ X.T @ (Y-Y_pred)


def _sigmoid(X):
    return 1/(1 + np.exp(-X))

class LogisticRegression:

    def __init__(self, fit_intercept, penalty, gamma) -> None:
        
        self.fit_intercept = fit_intercept
        self.beta = None
        self.gamma = None # gamma is the penalty factor
        self.penalty = penalty

    
    def predict(self, X): # correction

        #return _sigmoid(X @ self.beta / self.gamma)
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        return _sigmoid( X @ self.beta)

    def _NLL(self, X, Y, Y_pred):
        #Y_pred = self.predict(X)
        #Y_pred[Y == 1] + (1 - Y_pred)[ Y == 0]

        N, M = X.shape[0], X.shape[1]
        beta, gamma = self.beta, self.gamma

        order = 2 if self.penalty == "l2" else 1
        norm_beta = np.linalg.norm(beta, order=order)
        nll = np.log(Y_pred[Y==1]).sum() + np.log(1 - Y_pred[Y == 0]).sum()
        penalty = (gamma / 2) * norm_beta **2 if order == 2 else gamma * norm_beta

        return (nll + penalty) / N

    def _NLL_grad(self, X, Y, Y_pred):
        N, M = X.shape[0], X.shape[1]
        beta, gamma = self.beta, self.gamma

        order = 2 if self.penalty == "l2" else 1
        norm_beta = np.linalg.norm(beta, order=order)

        d_penalty = gamma * norm_beta if order == 2 else gamma * np.sign(beta)

        # d_sigmoid = (1 - Y_pred) * Y_pred @ X ???
        d_sigmoid = (Y - Y_pred) @ X

        return -(d_sigmoid + d_penalty) /N

    def fit(self, X, Y):

        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        i = 0
        """while i < max_iter:
            Y_pred = self.predict(X)
            self.beta += lr * self._NLL_grad(X, Y, Y_pred)"""

        while i< max_iter:
            Y_pred = _sigmoid(X @ self.beta)
            loss  = self._NLL(X, Y, Y_pred)
            if np.abs(loss - l_prev) < tol:
                return
            l_prev = loss
            self.beta -= lr * self._NLL_grad(X, Y, Y_pred)



class KNN:

    def __init__(self, K, X, Y) -> None:

        self.K = K
        self.X = X
        self.Y = Y

    def fit(self, X, Y):
        pass

    def predict(self, x):

        dists = np.linalg.norm(self.X - x, order = 2)
        idx_ranked = np.argsort(dists)
        counts = np.bincount(self.Y[idx_ranked[:self.K]])
        return np.argmax(counts)


        





    
    