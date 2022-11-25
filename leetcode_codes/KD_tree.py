import math
import numpy as np
from binarytree import tree, Node

# pandas version: https://www.youtube.com/watch?v=HWzyEslmSOg&t=113s

class generalNode:
    def __init__(self, val, left, right) -> None:
        self.val = val
        self.left = left
        self.right = right

class KDNode(Node):
    def __init__(self, _id, data, idx=0, left = None, right = None) -> None:
        super().__init__(_id)
        #self.value = _id
        self.data = data # value is a tuple or list
        self.idx = idx
        self.left = left
        self.right = right

class KDTree:

    def __init__(self, data) -> None:
        self.data = data
        self.root = None
        self.K = len(data[0]) if len(data) else 0

    def chooseBest(self, target, node1, node2):
        dist1 = self.dist(target, node1)
        dist2 = self.dist(target, node2)
        
        if dist1 <= dist2:
            return node1, dist1
        else:
            return node2, dist2

    def dist(self, target, node):
        if not node:
            return float("inf")

        return sum((i-j)**2 for i, j in zip(target, node.data))

    def find_NN(self, node, target, depth=0):
        
        if node is None:
            return None, float("inf")

        K = self.K
        idx = depth % K
        if target[idx] > node.data[idx]:
            next = node.right
            drop = node.left
        else:
            next = node.left
            drop = node.right

        sub, sub_dist = self.find_NN(next, target, depth + 1)
        best, best_dist = self.chooseBest(target, sub, node)

        if target[idx] - node.data[idx] < best_dist:
            sub2, sub_dist = self.find_NN(drop, target, depth + 1)
            best, best_dist = self.chooseBest(target, best, sub2)

        return best, best_dist

    def _build(self, data, depth, K):

        L = len(data)
        if L == 0:
            return None
        mid = L // 2
        data.sort(axis = depth%K) #data.sort(key = lambda x:x[depth%K])
        ind = np.lexsort(data[:, depth%K],)
        # ind = np.argsort(data[:, depth%K])
        data = data[ind]

        """
            _col = data.columns[depth%k]
            obj_lst = data.sort_values(by = [_col], ascending=True)
            node = KDNode(obj_lst.iloc[mid], idx = mid)
            node.left = self._build(obj_lst.iloc[:mid], depth=depth+1)
            node.right = self._build(obj_lst.iloc[mid+1:], depth=depth+1)
        """

        node = KDNode(int(data[mid][depth%K]), data[mid], depth%K)
        node.left = self._build(data[:mid], depth+1, K)
        node.right = self._build(data[mid+1:], depth+1, K)

        return node

    def build(self):
        self.root = self._build(self.data, depth=0, K=len(self.data[0]))


def check(target, data):
    """
    Compute dists betwwen target and all the data points
    Parameters:
    -----------
    target: (M,)
    data: (N, M)
    Returns:
    --------
    lst_dists: (N, )
        List of distance between target and all the data points.
    """

    if isinstance(target, list):
        target = np.array(target, dtype = np.float16)

    return np.linalg.norm(target - data, axis = 1)


target = [6,7,3,9,2]
data = np.random.randint(0,20,[10,5])
# data = [[2,3, 5],[5,4,2],[4,7,0],[8,1,9],[9,6,7],[7,2,4]]
kd_tree = KDTree(data)
kd_tree.build()
print(kd_tree.root)

best, best_dist = kd_tree.find_NN(kd_tree.root, target)
res = check(target, data)
check_res = data[np.argmin(res, axis = 0)]

print("knn result matches checked result") if np.equal(check_res, best.data).all() else print("knn doesn't get the optimal result.")

