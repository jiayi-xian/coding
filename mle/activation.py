"""A collection of activation function objects for building neural networks"""
from math import erf
from abc import ABC, abstractmethod

import numpy as np


class ActivationBase(ABC):
    def __init__(self, **kwargs):
        """Initialize the ActivationBase object"""
        super().__init__()

    def __call__(self, z):
        """Apply the activation function to an input"""
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.fn(z)

    @abstractmethod
    def fn(self, z):
        """Apply the activation function to an input"""
        raise NotImplementedError

    @abstractmethod
    def grad(self, x, **kwargs):
        """Compute the gradient of the activation function wrt the input"""
        raise NotImplementedError

class Sigmoid(ActivationBase):
    def __init__(self):
        """A logistic sigmoid activation function."""
        super().__init__()

    def __str__(self):
        """Return a string representation of the activation function"""
        return "Sigmoid"

    def fn(self, z):
        r"""
        Evaluate the logistic sigmoid, :math:`\sigma`, on the elements of input `z`.
        .. math::
            \sigma(x_i) = \frac{1}{1 + e^{-x_i}}
        """
        return 1 / (1 + np.exp(-z))

    def grad(self, x):
        r"""
        Evaluate the first derivative of the logistic sigmoid on the elements of `x`.
        .. math::
            \frac{\partial \sigma}{\partial x_i} = \sigma(x_i) (1 - \sigma(x_i))
        """
        fn_x = self.fn(x)
        return fn_x * (1 - fn_x)

    def grad2(self, x):
        r"""
        Evaluate the second derivative of the logistic sigmoid on the elements of `x`.
        .. math::
            \frac{\partial^2 \sigma}{\partial x_i^2} =
                \frac{\partial \sigma}{\partial x_i} (1 - 2 \sigma(x_i))
        """
        fn_x = self.fn(x)
        return fn_x * (1 - fn_x) * (1 - 2 * fn_x)

class ReLU(ActivationBase):

    def __init__(self):
        super().__init__()

    def __str__(self):
        """Return a string representation of the activation function"""
        return "ReLU"

    def fn(self, z):
        return np.clip(z, 0, np.inf)

    def grad(self, x):
        r"""
        Evaulate the first derivative of the ReLU function on the elements of input `x`.
        .. math::
            \frac{\partial \text{ReLU}}{\partial x_i}
                &=  1 \ \ \ \ &&\text{if }x_i > 0 \\
                &=  0   \ \ \ \ &&\text{otherwise}
        """
        return (x > 0).astype(int) # x>0 yield boolean ndarray, astype change it into int ndarray

    def grad2(self, x):
        r"""
        Evaulate the second derivative of the ReLU function on the elements of
        input `x`.
        .. math::
            \frac{\partial^2 \text{ReLU}}{\partial x_i^2}  =  0
        """
        return np.zeros_like(x)

class LeakyReLU(ActivationBase):
    """
    'Leaky' version of a rectified linear unit (ReLU).
    Notes
    -----
    Leaky ReLUs [*]_ are designed to address the vanishing gradient problem in
    ReLUs by allowing a small non-zero gradient when `x` is negative.
    Parameters
    ----------
    alpha: float
        Activation slope when x < 0. Default is 0.3.
    References
    ----------
    .. [*] Mass, L. M., Hannun, A. Y, & Ng, A. Y. (2013). "Rectifier
       nonlinearities improve neural network acoustic models." *Proceedings of
       the 30th International Conference of Machine Learning, 30*.
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        super().__init__()

    def __str__(self):
        """Return a string representation of the activation function"""
        return "Leaky ReLU(alpha={})".format(self.alpha)

    def fn(self, z):
        r"""
        Evaluate the leaky ReLU function on the elements of input `z`.
        .. math::
            \text{LeakyReLU}(z_i)
                &=  z_i \ \ \ \ &&\text{if } z_i > 0 \\
                &=  \alpha z_i \ \ \ \ &&\text{otherwise}
        """
        _z = z.copy()
        _z[z < 0] = _z[z < 0] * self.alpha
        return _z

    def grad(self, x):
        r"""
        Evaluate the first derivative of the leaky ReLU function on the elements
        of input `x`.
        .. math::
            \frac{\partial \text{LeakyReLU}}{\partial x_i}
                &=  1 \ \ \ \ &&\text{if }x_i > 0 \\
                &=  \alpha \ \ \ \ &&\text{otherwise}
        """
        out = np.ones_like(x)
        out[x < 0] *= self.alpha
        return out

    def grad2(self, x):
        r"""
        Evaluate the second derivative of the leaky ReLU function on the
        elements of input `x`.
        .. math::
            \frac{\partial^2 \text{LeakyReLU}}{\partial x_i^2}  =  0
        """
        return np.zeros_like(x)

class Affine(ActivationBase): # 点乘 dot product

    def __init__(self, slope=1, intercept=0):
        """
        An affine activation function.
        Parameters
        ----------
        slope: float
            Activation slope. Default is 1.
        intercept: float
            Intercept/offset term. Default is 0.
        """
        self.slope = slope
        self.intercept = intercept
        super().__init__()

    def __str__(self):
        """Return a string representation of the activation function"""
        return "Affine(slope={}, intercept={})".format(self.slope, self.intercept)

    def fn(self, z):
        r"""
        Evaluate the Affine activation on the elements of input `z`.
        .. math::
            \text{Affine}(z_i)  =  \text{slope} \times z_i + \text{intercept}
        """
        return self.slope * z + self.intercept

    def grad(self, x):
        r"""
        Evaluate the first derivative of the Affine activation on the elements
        of input `x`.
        .. math::
            \frac{\partial \text{Affine}}{\partial x_i}  =  \text{slope}
        """
        return self.slope * np.ones_like(x)

    def grad2(self, x):
        r"""
        Evaluate the second derivative of the Affine activation on the elements
        of input `x`.
        .. math::
            \frac{\partial^2 \text{Affine}}{\partial x_i^2}  =  0
        """
        return np.zeros_like(x)

class Softmax(ActivationBase):

    def __init__(self):
        super().__init__()


    def __str__(self) -> str:
        """Return a string representation of the activation function"""
        return "Softmax"

    def fn(self, z):
        z_max = np.max(z, axis=1)
        return np.exp(z_max)/np.sum(np.exp(z), axis = 1)

    # https://automata88.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
    def grad(self,s, X):
        # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
        # input s is softmax value of the original input x. 
        # s.shape = (1, n) 
        # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])
        # initialize the 2-D jacobian matrix.
        s = self.fn(X)
        jacobian_m = np.diag(s)
        for i in range(len(jacobian_m)):
            for j in range(len(jacobian_m)):
                if i == j:
                    jacobian_m[i][j] = s[i] * (1-s[i])
                else: 
                    jacobian_m[i][j] = -s[i]*s[j]
        return jacobian_m

        return 0

softmax = Softmax()

z = np.array([[1,2,3,4,1], [1,1,1,1,1]])
res = softmax(z)
print("Done")


