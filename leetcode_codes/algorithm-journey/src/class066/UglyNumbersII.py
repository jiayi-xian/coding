from typing import *
from collections import *
import heapq
from itertools import *

"""
丑数 II
给你一个整数 n ，请你找出并返回第 n 个 丑数
丑数 就是只包含质因数 2、3 或 5 的正整数
测试链接 : https://leetcode.cn/problems/ugly-number-ii/

方法一: 从一开始 每个自然数都验证一下看是不是丑数
方法2: 一个丑数 一定是前面某个丑数乘以2, 3或5
方法3: 定义三个指针 乘235 指针存储它指向的值乘以235的值 当当前值*235得到的值 选做第x个丑数 那么该指针释放 并指向下一个丑数
当前新增的丑数是在三种可能性中取最小 (这种维持单调性的办法refer to 双指针问题)
"""

class Solution:
    def nthUglyNumber(self, n: int) -> int:

        dp = [0] * (n+1)
        dp[1] = 1

        k = 2
        plst = [1, 1, 1]
        base = [2, 3, 5]
        while k < n+1:
            A = [dp[p]*b for p, b in zip(plst, base)]
            minA = min(A)
            for i in range(3):
                plst[i] += 1 if  A[i] == minA else 0
            dp[k] = minA
            k += 1
        return dp[n]


if __name__ == "__main__":
    sol = Solution()
    n = 8
    res = sol.nthUglyNumber(n)
    print(res)