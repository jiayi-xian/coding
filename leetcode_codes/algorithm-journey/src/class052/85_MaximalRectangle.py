from typing import *
from collections import *
import heapq
from itertools import *


"""
最大矩形
给定一个仅包含 0 和 1 、大小为 rows * cols 的二维二进制矩阵
找出只包含 1 的最大矩形，并返回其面积
测试链接：https://leetcode.cn/problems/maximal-rectangle/
"""


class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:

        def maximalLength(arr):
            stack = [-1]
            res = 0
            arr.append(-1)
            for idx, h in enumerate(arr):
                
                while stack and arr[stack[-1]] > h:
                    c = stack.pop() # ! 忘记加这一行会导致死循环 
                    res = max(res, (idx - stack[-1]-1)*arr[c]) 
                    # ! 这样写一定要避免stack为空 注意双-1的情况 ! 一开始写成了(idx-c) *(c - stack[-1]) 的形式
                stack.append(idx)
            arr.pop()
            return res
        
        arr = [0] * len(matrix[0])
        ans = 0
        for j, row in enumerate(matrix):
            for i, v in enumerate(row):
                if v == "0":
                    arr[i] = 0
                else:
                    arr[i] += int(v)
            ans = max(ans, maximalLength(arr)) 
        
        return ans




if __name__ == "__main__":
    sol = Solution()
    matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
    res = sol.maximalRectangle(matrix)
    print(res)