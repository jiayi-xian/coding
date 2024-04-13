"""

暴力递归 O(2^n) 因为每次都展开两条path
暴力递归时间复杂度差的原因：重复展开 F(7) -> F(6) F(5) ,F(6) -> F(5), F(4) 此时再展开展开F(5)就重复计算了
但如果我们使用缓存表记录F(i) 那么只需要一条path就行 O(n)

菲波契那数列问题 最优解 O(logn) 矩阵快速幂
"""

def fib1(n):

    dp = [-1] * n

    def helper(n, dp):
        if n == 1:
            return 1
        if n == 0:
            return 0
        
        if dp[n] != -1:
            return dp[n]
        else:
            dp[n] =  helper(n-1, dp) + helper(n-2, dp)
            return dp[n]

def fib1(n):

    dp = [-1] * n

    dp[0], dp[1] = 0, 1
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

def fib2(n):

    a, b = 0, 1
    for i in range(2, n+1):
        c = a + b
        a, b = b, c
    
    return c
    
