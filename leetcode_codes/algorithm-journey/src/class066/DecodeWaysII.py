from typing import *
from collections import *
import heapq
from itertools import *

"""
解码方法 II
一条包含字母 A-Z 的消息通过以下的方式进行了 编码 ：
'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
要 解码 一条已编码的消息，所有的数字都必须分组
然后按原来的编码方案反向映射回字母（可能存在多种方式）
例如，"11106" 可以映射为："AAJF"、"KJF"
注意，像 (1 11 06) 这样的分组是无效的，"06"不可以映射为'F'
除了上面描述的数字字母映射方案，编码消息中可能包含 '*' 字符
可以表示从 '1' 到 '9' 的任一数字（不包括 '0'）
例如，"1*" 可以表示 "11"、"12"、"13"、"14"、"15"、"16"、"17"、"18" 或 "19"
对 "1*" 进行解码，相当于解码该字符串可以表示的任何编码消息
给你一个字符串 s ，由数字和 '*' 字符组成，返回 解码 该字符串的方法 数目
由于答案数目可能非常大，返回10^9 + 7的模
测试链接 : https://leetcode.cn/problems/decode-ways-ii/
"""

class Solution_:
    def numDecodings(self, s: str):

        # 递归参数：index i s[i] letter 
        # options: 0，个位数，两位数（《=26 还有 if * )

        def recur(i, pre):

            if i >= len(s):
                if pre == "":
                    return 1
                else:
                    return 0

            c = s[i]
            # if c == "0":
            #    return 0
            
            res = 0 # 注意c就是*的情况
            if c == "*":
                for v in range(1,10):
                    if pre == "":
                        res += recur(i+1, "")
                    else:
                        res += recur(i+1, "") if int(pre+str(v)) <= 26 else 0
                    if 0<v<=2:
                        res += recur(i+1, str(v))
            else:
                if pre == "":
                    # c is decoded individually
                    res += recur(i+1, "") if c != "0" else 0
                    # c is decoded with pre
                else:
                    res += recur(i+1, "") if int(pre+c) <= 26 else 0
                if 0 < int(c) <= 2:
                    res += recur(i+1, c)
            return res
        ans = recur(0, "")
        return ans
    
    def numDecodings_memo(self, s: str):

        # 递归参数：index i s[i] letter 
        # options: 0，个位数，两位数（《=26 还有 if * )

        def recur(i, pre, dp):

            if i >= len(s):
                if pre == "":
                    return 1
                else:
                    return 0

            if dp[i] is not None:
                return dp[i]
            c = s[i]
            # if c == "0":
            #    return 0
            
            res = 0 # 注意c就是*的情况
            if c == "*":
                for v in range(1,10):
                    if pre == "":
                        res += recur(i+1, "", dp)
                    else:
                        res += recur(i+1, "", dp) if int(pre+str(v)) <= 26 else 0
                    if 0<v<=2:
                        res += recur(i+1, str(v), dp)
            else:
                if pre == "":
                    # c is decoded individually
                    res += recur(i+1, "", dp) if c != "0" else 0
                    # c is decoded with pre
                else:
                    res += recur(i+1, "", dp) if int(pre+c) <= 26 else 0
                if 0 < int(c) <= 2:
                    res += recur(i+1, c, dp)
            dp[i] = res
            return res
        dp = [None] * len(s)
        ans = recur(0, "", dp)
        return ans


class Solution:
    def numDecodings(self, s: str):

        # 递归参数：index i s[i] letter 
        # options: 0，个位数，两位数（《=26 还有 if * )
        def recur(i):

            if i >= len(s):
                return 1

            c = s[i]
            res = 0
            if c == "0":
                return 0
            # c is not *
            if c != "*":
                # decode c
                res += recur(i+1) if c != "0" else 0
                if i < len(s) -1: # decode s[i:i+2]
                    nc = s[i+1]
                    if nc != "*" and int(s[i:i+2])<=26:
                        res += recur(i+2)
                    else:
                        res += 9*recur(i+2)*(c=="1") + 6*recur(i+2) * (c=="2")
            else:
                # decode c
                res += recur(i+1)*9
                if i<len(s)-1: # decode * nc
                    nc = s[i+1]
                    if nc != "*": # ! 糊涂了
                        res += recur(i+2) *2 if int(nc) <= 6 else recur(i+2)
                    else: # decode * *
                        res += recur(i+2)*9 + recur(i+2)*6
            return res
        
        ans = recur(0)
        return ans
    
    def numDecodings_memo(self, s: str):

        # 递归参数：index i s[i] letter 
        # options: 0，个位数，两位数（《=26 还有 if * )
        def recur(i, dp):

            if i >= len(s): # 相当于 dp[n] = 1
                return 1

            if dp[i] != -1:
                return dp[i]
            c = s[i]
            res = 0
            if c == "0":
                return 0
            # c is not *
            if c != "*":
                # decode c
                res += recur(i+1, dp) if c != "0" else 0
                if i < len(s) -1: # decode s[i:i+2]
                    nc = s[i+1]
                    # if nc != "*" and int(s[i:i+2])<=26: # ! 必须是对 nc == "*" 的if else
                    if nc != "*":
                        res += recur(i+2, dp) if int(s[i:i+2])<=26 else 0
                    else:
                        # ! res += 9*recur(i+2, dp) if c=="1" else 6*recur(i+2, dp)
                        res += 9*(c=="1")*recur(i+2, dp) + 6*(c=="2")*recur(i+2, dp)
            else:
                # decode c
                res += recur(i+1, dp)*9
                if i<len(s)-1: # decode * nc
                    nc = s[i+1]
                    if nc != "*": 
                        # ! res += recur(i+2, dp) if int(c+nc) <= 26 else 0
                        res += recur(i+2, dp) *2 if int(nc) <= 6 else recur(i+2, dp)
                    else: # decode * *
                        res += recur(i+2, dp)*9 + recur(i+2, dp)*6
            dp[i] = res
            return res % M
        
        dp = [-1 ] * len(s)
        M = 10**9+7
        ans = recur(0, dp)

        return ans % (10**9 + 7)

    def numDecodings_dp(self, s):

        dp = [-1] * (len(s)+1)

        dp[len(s)] = 1
        n = len(s)
        for i in range(n-1, -1, -1):
            if s[i] == "0": 
                dp[i] = 0
                continue
            dp[i] = 0
            if s[i] != "*":
                dp[i] += dp[i+1]
                if i < len(s)-1:
                    if s[i+1] != "*":
                        dp[i] += dp[i+2] if int(s[i: i+2]) <= 26 else 0
                    else:
                        dp[i] += 9*dp[i+2] if s[i] == "1" else 0
                        dp[i] += 6*dp[i+2] if s[i] == "2" else 0
            elif s[i] == "*":
                dp[i] += 9*dp[i+1]
                if i < len(s)-1:
                    if s[i+1] != "*":
                        dp[i] += 2* dp[i+2] if int(s[i+1]) <= 6 else 1* dp[i+2]
                    else:
                        dp[i] += 15*dp[i+2]
        
        return dp[0]
    
    def numDecodings_dpc(self, s):
        n = len(s) 
        # 转移方程 dp[i] = f(dp[i+1] i 单独decode, dp[i+2] i i+1 一起decode)
        cur, nex, nnex = 0, 1, 1
        M = 10**9+7

        for i in range(n-1, -1, -1):
            if s[i] == "0": 
                cur = 0
                cur, nex, nnex = 0, cur % M, nex % M # ! countinue 之间不要忘记移动
                continue
            cur += 9*nex if s[i]=="*" else 1*nex # i decode
            if s[i] != "*":
                if i<n-1:
                    if s[i+1] != "*":
                        cur += nnex if int(s[i:i+2]) <= 26 else 0
                    else:
                        if s[i] == "1":
                            cur += 9*nnex
                        elif s[i] == "2":
                            cur += 6*nnex 
            else:
                if i<n-1:
                    if s[i+1] != "*":
                        cur += nnex *2 if int(s[i+1]) <= 6 else nnex
                    else:
                        cur += nnex * 15
            cur, nex, nnex = 0, cur % M, nex % M
        return nex 

        # 涉及到的用于存储的变量


if __name__ == "__main__":
    sol = Solution()
    s = "3*"
    res = sol.numDecodings_dp_compressed(s)
    print(res)