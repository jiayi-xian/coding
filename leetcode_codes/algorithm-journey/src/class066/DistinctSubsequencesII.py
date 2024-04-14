from typing import *
from collections import *
import heapq
from itertools import *


"""
不同的子序列 II
给定一个字符串 s，计算 s 的 不同非空子序列 的个数
因为结果可能很大，所以返回答案需要对 10^9 + 7 取余
字符串的 子序列 是经由原字符串删除一些（也可能不删除）
字符但不改变剩余字符相对位置的一个新字符串
例如，"ace" 是 "abcde" 的一个子序列，但 "aec" 不是
测试链接 : https://leetcode.cn/problems/distinct-subsequences-ii/
"""


class Solution:

    def distinctSubseqII(self, s: str) -> int:
        
        pass


    def distinct_subseq_ii(s: str) -> int:
        mod = 1000000007
        cnt = [0] * 26
        all = 1
        for x in s:
            new_add = (all - cnt[ord(x) - ord('a')] + mod) % mod
            cnt[ord(x) - ord('a')] = (cnt[ord(x) - ord('a')] + new_add) % mod
            all = (all + new_add) % mod
        return (all - 1 + mod) % mod





if __name__ == "__main__":
    sol = Solution()

    res = sol.distinctSubseqII(s)
    print(res)