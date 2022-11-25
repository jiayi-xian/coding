from abc import ABC, abstractmethod
from activation import Sigmoid
import numpy as np
class LayerBase(ABC):
    def __init__(self, optimizer=None):
        """An abstract base class inherited by all neural network layers"""
        self.X = []
        self.act_fn = None
        self.trainable = True
        #self.optimizer = OptimizerInitializer(optimizer)()

        self.gradients = {}
        self.parameters = {}
        self.derived_variables = {}

        super().__init__()

    #@abstractmethod
    def _init_params(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, z, **kwargs):
        """Perform a forward pass through the layer"""
        raise NotImplementedError

    @abstractmethod
    def backward(self, out, **kwargs):
        """Perform a backward pass through the layer"""
        raise NotImplementedError


class FullyConnected(LayerBase):
    def __init__(self, n_out, act_fn=None, init="glorot_uniform", optimizer=None):

        self.init = init
        self.n_in = None
        self.n_out = n_out
        #self.act_fn = ActivationInitializer(act_fn)()
        self.act_fn = act_fn
        self.parameters = {"W1": None, "b1": None, "W2": None, "b2": None}
        self.is_initialized = False

    def forward(self, X):
        """
        
        Parameters:
        -----------
        Wi: (M_i, M_{i+1})
        bi: (M_i)
        Returns:
        --------
        
        """
        W1, b1, W2, b2 = self.parameters["W1"],  self.parameters["b1"], self.parameters["W2"], self.parameters["b2"]
        # First layer pre-activation
        z1 = X @ W1 + b1 # (N, M)

        # First layer activation
        a1 = self.act_fn(z1)

        # Second layer pre-activation
        z2 = a1 @ W2 + b2

        a2 = self.act_fn(z2)


    def backward(self, dLdy, retain_grads=True):
        """
        Backprop from layer outputs to inputs.
        Parameters
        ----------
        dLdy : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)` or list of arrays
            The gradient(s) of the loss wrt. the layer output(s).
        Returns
        -------
        dLdX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)` or list of arrays
            The gradient of the loss wrt. the layer input(s) `X`.
        """  # noqa: E501
        assert self.trainable, "Layer is frozen"
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        dX = []
        X = self.X
        for dy, x in zip(dLdy, X):
            dx, dw, db = self._bwd(dy, x)
            dX.append(dx)

            if retain_grads:
                self.gradients["W"] += dw
                self.gradients["b"] += db

        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLdy, X):
        """Actual computation of gradient of the loss wrt. X, W, and b"""
        W = self.parameters["W1"]
        b = self.parameters["b1"]

        Z = X @ W + b
        dZ = dLdy * self.act_fn.grad(Z) # (N, d_out) * (N, d_out)

        dX = dZ @ W.T   #(N, d_out) @ (d_out, d_in)
        dW = X.T @ dZ   #(d_in, N) @ (N, d_out)
        dB = dZ.sum(axis=0, keepdims=True)  # (1, d_out)
        return dX, dW, dB



sig = Sigmoid()

# n_in: 5, n_out: 3
W1 = [[1,1,1,1,1], [2,2,2,2,2], [0,0,0,0,0]]
W2 = [[1,0,0], [0,1,0], [0,0,1]]
X1 = np.array([[1,2,3,4,5], [-1,-2,-3,-4,-5]])
X2 = np.array([-1,-2,-3,-4,-5])


b1, b2 = 0, 0
n_out = 3
nn = FullyConnected(n_out)
nn.parameters["W1"] = np.array(W1).T
nn.parameters["b1"] = np.array(b1)
nn.parameters["W2"] = np.array(W2).T
nn.parameters["b2"] = np.array(b2)
nn.act_fn = sig
res1 = nn.forward(X1)
res2 = nn._bwd(np.array([[0.2,0.2,0.2], [0.1,0.1,0.1]]), X1)




def sigmoid(x):
    return 1/(1 + np.exp(-x))

a = sigmoid(0.45+1.38)
print(a)
