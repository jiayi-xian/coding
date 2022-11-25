
import numpy as np
from metrics import dt_entropy, dt_gini_impurity, dt_mse
class Node:
    def __init__(self, rule, left=None, right=None) -> None:
        self.feature = rule[0]
        self.treshold = rule[1]
        self.left = left
        self.right = right

class Leaf:
    def __init__(self, value) -> None:
        self.value = value


class DecisionTree:
    def __init__(self, 
        classifier=True, 
        max_depth=None, 
        n_feats=None, 
        criterion="entropy",
        seed = None) -> None:
        """
        A decision tree model for regression and classification problem
        
        Parameters:
        -----------
        classifier: bool
            regression or classification
        max_depth: int
            the depth at which to stop growing the tree. If None, grow the tree until all leaves are pure.
        n_feats: int
            Specifies the number of features to sample on each split. If None, use all features on each split. Default is None.
        criterion: {"mse", "entropy", "gini"}
            The error criterion to use when calculating splits. 
        seed: int or None
            Seed for the random number generator. Default is None.
        
        Returns:
        --------
        """
        if seed:
            np.random.seed(seed)
        
        self.depth = 0
        self.root = None
        self.n_feats = n_feats
        self.criterion = criterion
        self.classifier = classifier
        self.max_depth = max_depth if max_depth else np.inf
    
    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def predict_class_probs(self, X):
        return np.array([self._traverse(x, self.root, prob=True) for x in X])

    def fit(self, X, Y):
        """
        Fit a binary decision tree to a dataset.
        Parameters:
        -----------
        X: (N, M)
        Y: (N,)
        Returns:
        --------
        """
        self.n_classes = max(Y) + 1
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self._grow(X, Y)

    def _grow(self, X, Y, cur_depth = 0):
        
        # if all labels are the same, return a leaf
        if len(set(Y)) == 1:
            if self.classifier:
                prob = np.zeros(self.n_classes)
                prob[Y[0]] = 1.0
            
            return Leaf(prob) if self.classifier else Leaf(Y[0])
            

        # if we have reached max depth, return a leaf
        if cur_depth >= self.max_depth:
            v = np.mean(Y)
            if self.classifier:
                v = np.bincount(Y, minlength=self.n_classes) / len(Y)
            return Leaf(v) # In the leaf node, there is a probability that the input classified into any class


        cur_depth += 1
        self.depth = max(cur_depth, self.depth)
        
        N, M = X.shape
        feat_idxs = np.random.choice(M, self.n_feats, replace=False)
        
        # select the best split according to 'criterion'
        feat, thresh = self._segment(X, Y, feat_idxs)
        l = np.argwhere(X[:, feat]<=thresh).flatten()
        r = np.argwhere(X[:, feat]>thresh).flatten()

        # grow the children that result from the split
        left = self._grow(X[l, :], Y[l], cur_depth) # left = self._grow(X[:, l], Y[l], cur_depth)
        right = self._grow(X[r, :], Y[r], cur_depth)

        return Node((feat, thresh), left, right)
    
    def _segment(self, X, Y, feat_idxs):
        """
        Find the optimal split rule (feature index and spliting threshold) according to self.criterion
        Parameters:
        -----------
        X: (N, M)
        Y: (N,)
        Returns:
        --------
        split_idx
        split_thresh
        """
        best_gain, split_idx = -np.inf, feat_idxs[0]
        for i in  feat_idxs:
            vals = X[:, i]
            levels = np.unique(vals)
            thresholds = (levels[1:] + levels[:-1])/2 if len(levels>1) else levels #[1,2,3,4] -> ([1,2,3] + [2,3,4]) /2 = [1.5, 2.5, 3.5]
            gains = np.array([self._impurity_gain(Y, thresh, X[:, i]) for thresh in thresholds])
        
            best_cur = gains.max()
            if best_cur > best_gain:
                best_gain = best_cur
                split_thresh = thresholds[ np.argmax(gains)]
                split_idx = i
        
        return split_idx, split_thresh

    def _impurity_gain(self, Y, split_thresh, feat_values):
        """
        compute impurity gain for input feature and input split threshold
        Parameters:
        -----------
        Y: (N,)
            label vector for N samples
        split_thresh: (1,)
            levels to split feature values
        feat_values: (K, )
            all feature values
        Returns:
        --------
        ig: np.float
            impurity gain attained from 
        """
        if self.criterion == "mse":
            loss = dt_mse
        if self.criterion == "gini":
            loss = dt_gini_impurity
        if self.criterion == "entropy":
            loss = dt_entropy # 是entropy不是cross entropy
        
        parent_loss = loss(Y)

        left = np.argwhere(feat_values <= split_thresh).flatten()
        right = np.argwhere(feat_values > split_thresh).flatten()

        if len(left) == 0 or len(right) == 0:
            return 0

        n = len(Y)
        n_l, n_r = len(left), len(right)
        e_l, e_r = loss(Y[left]), loss(Y[right])
        child_loss = (n_l/n) * e_l + (n_r/n) * e_r

        ig = parent_loss - child_loss

        return ig

    def _traverse(self, X, node, prob = False):

        if isinstance(node, Leaf):
            if self.classifier:
                #idx = np.argmax(np.random.choice(len(node.value), 1, node.value))
                return node.value if prob else node.value.argmax()
            return node.value


        if X[node.feature] <= node.threshold:
            return self._traverse(X, node.left, prob)
        else:
            return self._traverse(X, node.right, prob)




X = np.random.rand(7,3)
thresholds = np.mean(X, axis = 0)
l = np.argwhere(X[:, 0]<=thresholds[0]) # l.shape: (3, 1) [[1], [3], [6]] ndarray


4
print("Done")
