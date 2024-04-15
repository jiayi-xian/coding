"""
最低票价 
在一个火车旅行很受欢迎的国度，你提前一年计划了一些火车旅行
在接下来的一年里，你要旅行的日子将以一个名为 days 的数组给出
每一项是一个从 1 到 365 的整数
火车票有 三种不同的销售方式
一张 为期1天 的通行证售价为 costs[0] 美元
一张 为期7天 的通行证售价为 costs[1] 美元
一张 为期30天 的通行证售价为 costs[2] 美元
通行证允许数天无限制的旅行
例如，如果我们在第 2 天获得一张 为期 7 天 的通行证
那么我们可以连着旅行 7 天(第2~8天)
返回 你想要完成在给定的列表 days 中列出的每一天的旅行所需要的最低消费
测试链接 : https://leetcode.cn/problems/minimum-cost-for-tickets/
"""
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