class Code02_MinimumCostForTickets:
    durations = [1, 7, 30]

    # 暴力尝试
    def mincostTickets1(self, days, costs):
        return self.f1(days, costs, 0)

    def f1(self, days, costs, i):
        if i == len(days):
            return 0
        ans = float('inf')
        for k in range(3):
            j = i
            while j < len(days) and days[i] + self.durations[k] > days[j]:
                j += 1
            ans = min(ans, costs[k] + self.f1(days, costs, j))
        return ans

    # 暴力尝试改记忆化搜索
    # 从顶到底的动态规划
    def mincostTickets2(self, days, costs):
        dp = [float('inf')] * len(days)
        return self.f2(days, costs, 0, dp)

    def f2(self, days, costs, i, dp):
        if i == len(days):
            return 0
        if dp[i] != float('inf'):
            return dp[i]
        ans = float('inf')
        for k in range(3):
            j = i
            while j < len(days) and days[i] + self.durations[k] > days[j]:
                j += 1
            ans = min(ans, costs[k] + self.f2(days, costs, j, dp))
        dp[i] = ans
        return ans

    # 严格位置依赖的动态规划
    # 从底到顶的动态规划
    MAXN = 366

    dp = [float('inf')] * MAXN

    def mincostTickets3(self, days, costs):
        n = len(days)
        self.dp[:n + 1] = [float('inf')] * (n + 1)
        self.dp[n] = 0
        for i in range(n - 1, -1, -1):
            for k in range(3):
                j = i
                while j < len(days) and days[i] + self.durations[k] > days[j]:
                    j += 1
                self.dp[i] = min(self.dp[i], costs[k] + self.dp[j])
        return self.dp[0]