"""A collection of activation function objects for building neural networks"""
from math import erf
from abc import ABC, abstractmethod

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

    @abstractmethod
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

    def freeze(self):
        """
        Freeze the layer parameters at their current values so they can no
        longer be updated.
        """
        self.trainable = False

    def unfreeze(self):
        """Unfreeze the layer parameters so they can be updated."""
        self.trainable = True

    def flush_gradients(self):
        """Erase all the layer's derived variables and gradients."""
        assert self.trainable, "Layer is frozen"
        self.X = []
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = []

        for k, v in self.gradients.items():
            self.gradients[k] = np.zeros_like(v)

    def update(self, cur_loss=None):
        """
        Update the layer parameters using the accrued gradients and layer
        optimizer. Flush all gradients once the update is complete.
        """
        assert self.trainable, "Layer is frozen"
        self.optimizer.step()
        for k, v in self.gradients.items():
            if k in self.parameters:
                self.parameters[k] = self.optimizer(self.parameters[k], v, k, cur_loss)
        self.flush_gradients()

    def set_params(self, summary_dict):
        """
        Set the layer parameters from a dictionary of values.

        Parameters
        ----------
        summary_dict : dict
            A dictionary of layer parameters and hyperparameters. If a required
            parameter or hyperparameter is not included within `summary_dict`,
            this method will use the value in the current layer's
            :meth:`summary` method.

        Returns
        -------
        layer : :doc:`Layer <numpy_ml.neural_nets.layers>` object
            The newly-initialized layer.
        """
        layer, sd = self, summary_dict

        # collapse `parameters` and `hyperparameters` nested dicts into a single
        # merged dictionary
        flatten_keys = ["parameters", "hyperparameters"]
        for k in flatten_keys:
            if k in sd:
                entry = sd[k]
                sd.update(entry)
                del sd[k]

        for k, v in sd.items():
            if k in self.parameters:
                layer.parameters[k] = v
            if k in self.hyperparameters:
                if k == "act_fn":
                    layer.act_fn = ActivationInitializer(v)()
                elif k == "optimizer":
                    layer.optimizer = OptimizerInitializer(sd[k])()
                elif k == "wrappers":
                    layer = init_wrappers(layer, sd[k])
                elif k not in ["wrappers", "optimizer"]:
                    setattr(layer, k, v)
        return layer

    def summary(self):
        """Return a dict of the layer parameters, hyperparameters, and ID."""
        return {
            "layer": self.hyperparameters["layer"],
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
        }

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
        return np.clip(z, 0, np.inf) # np.max(0, z) # TODO ?

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

class Softmax_(ActivationBase):

    def __init__(self):
        super().__init__()


    def __str__(self) -> str:
        """Return a string representation of the activation function"""
        return "Softmax"

    def fn(self, z):
        z_max = np.max(z, axis=1).reshape(-1, 1)
        z -= z_max
        return np.exp(z)/np.sum(np.exp(z), axis = -1).reshape(-1,1)

    # https://automata88.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
    def grad(self, X):
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

class Softmax(LayerBase):
    def __init__(self, dim=-1, optimizer=None):
        r"""
        A softmax nonlinearity layer.

        Notes
        -----
        This is implemented as a layer rather than an activation primarily
        because it requires retaining the layer input in order to compute the
        softmax gradients properly. In other words, in contrast to other
        simple activations, the softmax function and its gradient are not
        computed elementwise, and thus are more easily expressed as a layer.

        The softmax function computes:

        .. math::

            y_i = \frac{e^{x_i}}{\sum_j e^{x_j}}

        where :math:`x_i` is the `i` th element of input example **x**.

        Parameters
        ----------
        dim: int
            The dimension in `X` along which the softmax will be computed.
            Default is -1.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with
            default parameters. Default is None. Unused for this layer.

        Attributes
        ----------
        X : list
            Running list of inputs to the :meth:`forward <numpy_ml.neural_nets.LayerBase.forward>` method since the last call to :meth:`update <numpy_ml.neural_nets.LayerBase.update>`. Only updated if the `retain_derived` argument was set to True.
        gradients : dict
            Dictionary of loss gradients with regard to the layer parameters
        parameters : dict
            Dictionary of layer parameters
        hyperparameters : dict
            Dictionary of layer hyperparameters
        derived_variables : dict
            Dictionary of any intermediate values computed during
            forward/backward propagation.
        """  # noqa: E501
        super().__init__(optimizer)

        self.dim = dim
        self.n_in = None
        self.is_initialized = False

    def _init_params(self):
        self.gradients = {}
        self.parameters = {}
        self.derived_variables = {}
        self.is_initialized = True

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "SoftmaxLayer",
            "n_in": self.n_in,
            "n_out": self.n_in,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    def forward(self, X, retain_derived=True):
        """
        Compute the layer output on a single minibatch.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            Layer input, representing the `n_in`-dimensional features for a
            minibatch of `n_ex` examples.
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            is True.

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)`
            Layer output for each of the `n_ex` examples.
        """
        if not self.is_initialized:
            self.n_in = X.shape[1]
            self._init_params()

        Y = self._fwd(X)

        if retain_derived:
            self.X.append(X)

        return Y

    def _fwd(self, X):
        """Actual computation of softmax forward pass"""
        # center data to avoid overflow
        e_X = np.exp(X - np.max(X, axis=self.dim, keepdims=True))
        return e_X / e_X.sum(axis=self.dim, keepdims=True)

    def backward(self, dLdy, retain_grads=True):
        """
        Backprop from layer outputs to inputs.

        Parameters
        ----------
        dLdy : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)` or list of arrays
            The gradient(s) of the loss wrt. the layer output(s). `n_ex`: bsz
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.

        Returns
        -------
        dLdX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            The gradient of the loss wrt. the layer input `X`.
        """  # noqa: E501
        assert self.trainable, "Layer is frozen"
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        dX = []
        X = self.X
        for dy, x in zip(dLdy, X):
            dx = self._bwd(dy, x)
            dX.append(dx)

        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLdy, X):
        """
        Actual computation of the gradient of the loss wrt. the input X.

        The Jacobian, J, of the softmax for input x = [x1, ..., xn] is:
            J[i, j] =
                softmax(x_i)  * (1 - softmax(x_j))  if i = j
                -softmax(x_i) * softmax(x_j)        if i != j
            where
                x_n is input example n (ie., the n'th row in X)
        """
        dX = [] # X: (n_ex, n_in)
        for dy, x in zip(dLdy, X): # for each sample 
            dxi = []
            for dyi, xi in zip(*np.atleast_2d(dy, x)): # for each coordinate in each samplex yi = \frac{e^{x_i}}{∑_j e^{xji}}
                """
                np.atleast_2d()
                它用于确保输入的数组至少是二维的。如果输入数组的维度小于2，该函数会将其提升到二维。具体来说，如果输入是一个一维数组（形状为 (n,)），
                它将被转换成一个形状为 (n, 1) 的二维数组；如果输入是一个二维或更高维度的数组，它将保持不变。

                np.diagflat 是 NumPy 中的一个函数，它用于创建一个以给定数组为对角线元素的二维数组（对角矩阵）。这个函数有两个主要参数：

                v：用来构成对角线元素的一维数组。
                k（可选）：指定对角线的位置。默认为0，意味着主对角线。如果 k 为正数，对角线将在主对角线之上；如果为负数，则在主对角线之下。
                """
                yi = self._fwd(xi.reshape(1, -1)).reshape(-1, 1) # xi.reshape(1, -1) add bsz dimention (M,) -> (1,M) _fwd -> (1,M) -> reshape(-1, 1) -> (M,1)
                dyidxi = np.diagflat(yi) - yi @ yi.T # ? 
                # jacobian wrt. input sample xi # (n_k, n_k) k: number of output classes

                dxi.append(dyi @ dyidxi) # (M, )
            dX.append(dxi)
        return np.array(dX).reshape(*X.shape)


z = np.array([[1,2,3,4,1], [1,1,1,1,1], [1,1,-1,1,-1], [1,-2,3,-4,1]])

softm = Softmax()
softm.forward(z)
softm.backward(z)