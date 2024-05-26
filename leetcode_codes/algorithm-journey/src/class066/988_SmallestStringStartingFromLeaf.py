from typing import *
from collections import *
import heapq
from itertools import *


"""
988. Smallest String Starting From Leaf
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def smallestFromLeaf(self, root: Optional[TreeNode]) -> str:

        res = set()
        def dfs(node):

            if node is None:
                return [""]
            
            left = dfs(node.left)

            right = dfs(node.right)
            return [s+node.val for s in left + right]

        res = dfs(root).sort()
        return res[0]

    def smallestFromLeaf(self, root: Optional[TreeNode]) -> str: 
        res = []
        def dfs2(node, pre):
            if node is None:
                res.append(pre)
            
            dfs2(node.left, pre+node.val)
            dfs2(node.right, pre+node.val)
        
        res.sort()
        return res[0]





if __name__ == "__main__":
    sol = Solution()

    res = sol.
    print(res)