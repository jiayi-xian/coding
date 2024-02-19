import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def factorization_machine(X, Y, Vi, learning_rate=0.01, epochs=100):
    N, M = X.shape
    K = Vi.shape[0]

    # Initialize model parameters
    W_0 = 0.0
    W = np.zeros(M)
    V = np.random.randn(M, K)  # Initialize latent factors randomly

    for epoch in range(epochs):
        linear_term = W_0 + np.dot(X, W)
        interaction_term = 0.5 * np.sum(np.dot(X, V) ** 2 - np.dot(X ** 2, V ** 2), axis=1)
        # np.dot(X, V) -> 
        y_pred = linear_term + interaction_term
        error = y_pred - Y

        grad_W_0 = np.mean(error)
        grad_W = np.dot(X.T, error) / N
        grad_V = np.dot(X.T, (error[:, np.newaxis] * (np.dot(X, V) - X.dot(V ** 2))).reshape(N, M, K)) / N

        # Update model parameters using gradient descent
        W_0 -= learning_rate * grad_W_0
        W -= learning_rate * grad_W
        V -= learning_rate * grad_V

    return W_0, W, V

# Example usage
N = 1000  # Number of data points
M = 10   # Number of features
K = 5    # Number of latent factors

X = np.random.rand(N, M)
Y = np.random.rand(N)
Vi = np.random.rand(K)

W_0, W, V = factorization_machine(X, Y, Vi)
