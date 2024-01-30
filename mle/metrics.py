import numpy as np

def kmean_euclidean(point, data):
    """
    Parameters
    ----------
    point : :py:class:`ndarray <numpy.ndarray>` of shape `(M,)`
    data : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
    
    Returns
    -------
    list of Euclidean dists between point and each point in data
    """
    # np.sqrt(np.sum((point - data)**2, axis=1))
    return np.linalg.norm((data-point), ord=2, axis = 1)

def dt_mse(y):
    """
    compute mse for decision tree node
    Parameters:
    -----------
    
    Returns:
    --------
    
    """
    return np.mean((y-np.mean(y)))**2

def dt_entropy(y):
    """
    compute entropy of a sequence of labels.
    Parameters:
    -----------
    y: (N,)
        a sequence of labels (of samples classified into one node in DT)
    Returns:
    --------
    output: (1,)
    """
    hist = np.bincount(y)
    ps = hist/np.sum(hist)
    # return -np.sum(ps*np.log2(ps))
    return -np.sum([p*np.log2(p) for p in ps if p>0])

def dt_gini_impurity(y):
    """
    Gini impurity (local entropy) of a label sequence
    Parameters:
    -----------
    y: (N, )
        array of integer labels
    Returns:
    --------
    output: (1, )
        gini impurity of a data set
    """
    hist = np.bincount(y)
    N = np.sum(hist)
    return 1 - sum([[(i/N)**2 for i in hist]])

import numpy as np


def euclidean(x, y):
    """
    Compute the Euclidean (`L2`) distance between two real vectors

    Notes
    -----
    The Euclidean distance between two vectors **x** and **y** is

    .. math::

        d(\mathbf{x}, \mathbf{y}) = \sqrt{ \sum_i (x_i - y_i)^2  }

    Parameters
    ----------
    x,y : :py:class:`ndarray <numpy.ndarray>` s of shape `(N,)`
        The two vectors to compute the distance between

    Returns
    -------
    d : float
        The L2 distance between **x** and **y**.
    """
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan(x, y):
    """
    Compute the Manhattan (`L1`) distance between two real vectors

    Notes
    -----
    The Manhattan distance between two vectors **x** and **y** is

    .. math::

        d(\mathbf{x}, \mathbf{y}) = \sum_i |x_i - y_i|

    Parameters
    ----------
    x,y : :py:class:`ndarray <numpy.ndarray>` s of shape `(N,)`
        The two vectors to compute the distance between

    Returns
    -------
    d : float
        The L1 distance between **x** and **y**.
    """
    return np.sum(np.abs(x - y))


def chebyshev(x, y):
    """
    Compute the Chebyshev (:math:`L_\infty`) distance between two real vectors

    Notes
    -----
    The Chebyshev distance between two vectors **x** and **y** is

    .. math::

        d(\mathbf{x}, \mathbf{y}) = \max_i |x_i - y_i|

    Parameters
    ----------
    x,y : :py:class:`ndarray <numpy.ndarray>` s of shape `(N,)`
        The two vectors to compute the distance between

    Returns
    -------
    d : float
        The Chebyshev distance between **x** and **y**.
    """
    return np.max(np.abs(x - y))


def minkowski(x, y, p):
    """
    Compute the Minkowski-`p` distance between two real vectors.

    Notes
    -----
    The Minkowski-`p` distance between two vectors **x** and **y** is

    .. math::

        d(\mathbf{x}, \mathbf{y}) = \left( \sum_i |x_i - y_i|^p \\right)^{1/p}

    Parameters
    ----------
    x,y : :py:class:`ndarray <numpy.ndarray>` s of shape `(N,)`
        The two vectors to compute the distance between
    p : float > 1
        The parameter of the distance function. When `p = 1`, this is the `L1`
        distance, and when `p=2`, this is the `L2` distance. For `p < 1`,
        Minkowski-`p` does not satisfy the triangle inequality and hence is not
        a valid distance metric.

    Returns
    -------
    d : float
        The Minkowski-`p` distance between **x** and **y**.
    """
    return np.sum(np.abs(x - y) ** p) ** (1 / p)


def hamming(x, y):
    """
    Compute the Hamming distance between two integer-valued vectors.

    Notes
    -----
    The Hamming distance between two vectors **x** and **y** is

    .. math::

        d(\mathbf{x}, \mathbf{y}) = \\frac{1}{N} \sum_i \mathbb{1}_{x_i \\neq y_i}

    Parameters
    ----------
    x,y : :py:class:`ndarray <numpy.ndarray>` s of shape `(N,)`
        The two vectors to compute the distance between. Both vectors should be
        integer-valued.

    Returns
    -------
    d : float
        The Hamming distance between **x** and **y**.
    """
    return np.sum(x != y) / len(x)