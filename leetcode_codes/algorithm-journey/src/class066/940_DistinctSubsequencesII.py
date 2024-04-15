from typing import *
from collections import *
import heapq
from itertools import *


"""
不同的子序列 II
给定一个字符串 s, 计算 s 的 不同非空子序列 的个数
因为结果可能很大，所以返回答案需要对 10^9 + 7 取余
字符串的 子序列 是经由原字符串删除一些（也可能不删除）
字符但不改变剩余字符相对位置的一个新字符串
例如，"ace" 是 "abcde" 的一个子序列，但 "aec" 不是
测试链接 : https://leetcode.cn/problems/distinct-subsequences-ii/
"""


class Solution:

    def distinctSubseqII(self, s: str) -> int:
        """
        这道题的特点是
        1. 对于s[i] 考虑以s[i]结尾的distinct Subseq有多少个（但事实上，计算distinct seq的数量不绝对依赖于idx 而是和
        字母本身有关， 譬如 abcabc
        2. #! 如果我们知道以s[i-1]结尾的distinct subseq有多少个 那么就容易知道s[i]结尾的distinct subseq有多少个 因为后者
        #! 是前者的distinct subseq concat s[i]，然后再减去重复计算的 
        # ! 上面的分析不对。应该如果我们知道当前截止到s[i-1](注意 不一定要以s[i-1]结尾)的distinct subseq有多少个 
        # ! 那么就容易知道s[i]结尾的distinct subseq有多少个 因为后者
        # ! 是前者的distinct subseq concat s[i]，然后再减去重复计算的

        3. 譬如说abcdabc 在第一个c时计算的distinct subseq 也可以看做是第二个c时的distinct subseq (只需要认为这些sub seq的
        c是来自第二个的即可
        ! 所以在计算第二个c的distinct subseq 数目时, 重复的计算了所有这样的subseq 故应该减去这部分的数量
        ! dp[char] = all - dp[char] in which it has the same letter with s[i]]
        计算以第二个c结尾的distinct subseq 就是之前的所有distinc subseq (all) 末尾接c 所以是之前所有distinct subseq的个数(可以验证两两不同)
        两个sub seq 不同当且仅当除了最后一位共同是"c"时 不同
        计算纯新增的distinct subseq: all + (all - dp["c"])
        """
        dp = [0]*26
        mod = 10**9+7
        res = 1 # "" 字符串
        dp[ord(s[0]) - ord('a')] = 1
        for i in range(1, len(s)):
            c = s[i]           
            dp[ord(c) - ord("a")] = res % mod
            res += res % mod - dp[ord(c)-ord("a")] % mod

        return res % mod

            



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