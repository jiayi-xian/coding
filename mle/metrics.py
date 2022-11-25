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