from abc import ABC, abstractmethod

import numpy as np

class ObjectiveBase(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def grad(self, y_true, y_pred, **kwargs):
        pass



import numpy as np
from abc import ABC, abstractmethod

class ObjectiveBase(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def grad(self, y_true, y_pred, **kwargs):
        pass

# get one hot
def get_onehot(index, vocab):
    one_hot = np.zeros(len(vocab))
    one_hot[index] = 1
    return one_hot

def onehot_encoding(categories, labels):
    # https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/preprocessing/general.py

    cat2idx = {c: i for i, c in enumerate(categories)}
    idx2cat = {i: c for i, c in enumerate(categories)}

    N, C = len(labels), len(cat2idx)
    cols = np.array([cat2idx[c] for c in labels])

    Y = np.zeros((N, C))
    Y[np.arange(N), cols] = 1
    return Y

# cross entropy
def cross_entropy_1D(y_hat, labels):
    """
    get cross_entropy
    Parameters:
    -----------
    y_hat: (N,) output logit of model
    labels: (N,)
    
    Returns:
    --------
    y * log y_hat
    
    """
    return labels * np.log(y_hat)

def cross_entropy(y_pred, y):
    """
    Alternative for binary: 
        nll = -np.log(y_pred[y == 1]).sum() - np.log(1 - y_pred[y == 0]).sum()
    Parameters:
    -----------
    y_pred: (N, M)
    y: (N, M)
    Returns:
    --------
    
    """
    eps = np.finfo(float).eps
    # That line says “give me the smallest possible positive number that the float datatype can represent on my machine”. In the context of the solution code, it’s being used to avoiding taking the log of 0 while still representing an asymptotically small likelihood.
    return -np.sum(y*np.log(y_pred+eps))

def cross_entropy_grad(y, y_pred):
    """
    gradient w.r.t z and softmax: z -> softmax(z) -> crossentropy
    Parameters:
    -----------
    y: (N, M)
    y_pred: (N, M)
    Returns:
    --------
    """
    grad = y_pred - y
    return grad

# softmax
def softmax(z):

    sm = np.exp(z) / (np.sum(np.exp(z)))
    return sm

def grad(Z, dLd):
    """
    
    Parameters:
    -----------
    z : (N, K)
    Returns:
    --------
    https://automata88.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d

    https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/layers/layers.py


    """
    dPdZ = [] # for N samples
    for z in Z: # for one sample
        p = softmax(z) # for one sample x -> z -> p
        p.reshape(-1, 1) # 这样才可以用p@p.T
        dpdz = np.diag(p) - p @ p.T
        dPdZ.append(dpdz) # each item represent the Jacobian of dpdz relative to the i-th sample

    return np.array(dPdZ)

# get word embedding

# get score
a= 3
# get window text (left_context, right_context)



y = np.array([1,0,0,0,0])
x = np.random.rand(5)
print(y@x.T)
# 0.10550604886660397
x = x.reshape((-1,1))
y = y.reshape((-1,1))
print(y@x.T) # (5,5) matrix