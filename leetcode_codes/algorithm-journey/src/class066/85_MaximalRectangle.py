"""
最大矩形
给定一个仅包含 0 和 1 、大小为 rows * cols 的二维二进制矩阵
找出只包含 1 的最大矩形，并返回其面积
测试链接：https://leetcode.cn/problems/maximal-rectangle/
"""

from typing import *
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        
        M = matrix
        n, m = len(M), len(M[0])
        H0 = M[0]
        maxA = -1

        for r in range(n):

            if r == 0:
                H = H0

            for idx, h in enumerate(M[r]):
                if h == 0:
                    H[idx] = 0
                else:
                    H[idx] += 1

            # get the maxRect
            H.append(-1)
            stack = [-1]
            for i in range(m):
                
                while stack and H[stack[-1]] > H[i]:
                    j = stack.pop()
                    maxA = max(maxA, H[j]*(i-j)*(j-(stack[-1] if stack else -1)))
                stack.append(i)
            
        return maxA

if __name__ == "__main__":
    s = Solution()
    M = None
    res = s.maximalRectangle(matrix=M)
    print(res)