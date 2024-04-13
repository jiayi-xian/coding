"""
解码方法
一条包含字母 A-Z 的消息通过以下映射进行了 编码 ：
'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
要 解码 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）
例如，"11106" 可以映射为："AAJF"、"KJF"
注意，消息不能分组为(1 11 06)，因为 "06" 不能映射为 "F"
这是由于 "6" 和 "06" 在映射中并不等价
给你一个只含数字的 非空 字符串 s ，请计算并返回 解码 方法的 总数
题目数据保证答案肯定是一个 32位 的整数
测试链接 : https://leetcode.cn/problems/decode-ways/

"""

class Solution:
    def numDecodings_rec(self, s: str) -> int:
        n = len(s)
        n2c = {str(i+1):chr(ord("A")+i-1) for i in range(26)} # 注意看题 从1开始
        
        def rec(i):
            
            if i == n:
                return 1 #TODO
            
            res = 0
            for j in range(1,3):
                if s[i] != "0" and i+j<=n and int(s[i:i+j]) <= 26: # TODO out of index, 0 issue
                    res += rec(i+j)
            return res

        return rec(0)
    
    def numDecodings_memo(self, s: str) -> int:
        n = len(s)
        n2c = {str(i+1):chr(ord("A")+i-1) for i in range(26)} # 注意看题 从1开始
        dp = [-1]*(len(s)+1)
        dp[n] = 0
        def rec(i):
            
            if i == n:
                return 1 #TODO
            
            if dp[i]!=-1:
                return dp[i]
            
            res = 0
            for j in range(1,3):
                if s[i] != "0" and i+j<=n and int(s[i:i+j]) <= 26: # TODO out of index, 0 issue
                    res += rec(i+j)
            dp[i] = res
            return dp[i]

        rec(0)
        return dp[0]
    
    def numDecodings_dp(self, s: str) -> int:
        n = len(s)
        dp = [0]*(len(s)+1)
        dp[n] = 1 # 从recursion可以看出来这里是1 另一种想法是1计算的是s[i:i+j]+“”这种分解方式

        for i in range(n, -1, -1):
            for j in range(1,3):
                if i+j <= n and s[i]!= "0" and int(s[i:i+j]) <= 26:
                    dp[i] += dp[i+j]
        return dp[0]

st = "226"
#st = "06"
s = Solution()
res = s.numDecodings_dp(st)
print(res)

