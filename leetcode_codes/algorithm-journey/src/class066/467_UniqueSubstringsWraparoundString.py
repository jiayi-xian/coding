from typing import *
from collections import *
import heapq
from itertools import *


"""
环绕字符串中唯一的子字符串
定义字符串 base 为一个 "abcdefghijklmnopqrstuvwxyz" 无限环绕的字符串
所以 base 看起来是这样的：
"..zabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcd.."
给你一个字符串 s ，请你统计并返回 s 中有多少 不同非空子串 也在 base 中出现
测试链接 : https://leetcode.cn/problems/unique-substrings-in-wraparound-string/
"""


class Solution:
    def findSubstringInWraproundString(self, s: str) -> int:

        """
        number of sub sequence ended with s[i] : 2**(i),
        number of sub array ended with s[i]: i
        这题有意思的地方在于 
        base中 以某个特定字母, 譬如说以a结尾的 pattern都是一样的 ...uvwxyza
        如果在s中出现两次以a结尾的子字符串 譬如 zabpe...xvxyza
        那么在计算以第二个a结尾的unique子字符串数量 则一定会包含第一个a结尾的unique子字符串数量，
        因为它们只能遵循base中的pattern 此时谁能往左延伸得更长就会包含短的
        所以cnt[s[i]] = max(cnt[s[i]], L) L 是当前子字符串的值
        """
        res = 0
        cnt = defaultdict(int)
        cnt[s[0]] = 1
        L = 0
        for i in range(1, len(s)):
            if ord(s[i])- ord(s[i-1]) == 1 or (s[i] == "a" and s[i-1] == "z"):
                L += 1
            else:
                L = 1
            cnt[s[i]] = max(cnt[s[i]], L)
        return sum( val for val in cnt.values())
        
        
        res = 0
        cnt = [0] * 26
        for i, c in enumerate(s):
            if i >0 and (ord(c) - ord(s[i-1]) == 1 or (s[i-1]=="z" and c == "a")):
                cnt[ord(c)-ord("a")] = cnt[s[i-1]]
            else:
                cnt[c] = 1
            res += cnt[c]
        return res


if __name__ == "__main__":
    sol = Solution()
    s = "xyzabwcdabc"
    # res = sol.findSubstringInWraproundString(s)
    res = sol.find_substring_in_wrapround_string(s)
    print(res)