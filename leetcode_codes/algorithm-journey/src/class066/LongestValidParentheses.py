from typing import *
from collections import *
import heapq
from itertools import *


"""
最长有效括号
给你一个只包含 '(' 和 ')' 的字符串
找出最长有效（格式正确且连续）括号子串的长度。
测试链接 : https://leetcode.cn/problems/longest-valid-parentheses/

i:以第i个字符结尾 最长的有效括号长度是多少
"""


class Solution:
    def longestValidParentheses(self, s: str) -> int:
        
        n = len(s)
        dp = [0] * n
        dp[0] = 0
        for i in range(1, n):
            if s[i] == "(":
                dp[i] = 0
                continue
            else:
                if i - dp[i-1] - 1>=0:
                    dp[i] = dp[i] + dp[i-1] + 2 if s[i-dp[i-1]-1] == "(" else 0
                    if i - dp[i-1] -2 >= 0:
                        dp[i] += dp[i-dp[i-1]-2]
        
        return max(dp)




if __name__ == "__main__":
    sol = Solution()
    s = "(()"
    res = sol.longestValidParentheses(s)
    print(res)