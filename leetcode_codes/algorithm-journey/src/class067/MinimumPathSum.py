"""
最小路径和
给定一个包含非负整数的 m x n 网格 grid
请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
说明：每次只能向下或者向右移动一步。
测试链接 : https://leetcode.cn/problems/minimum-path-sum/
"""

class Code01_MinimumPathSum:
    # 暴力递归
    def minPathSum1(self, grid):
        return self.f1(grid, len(grid) - 1, len(grid[0]) - 1)

    def f1(self, grid, i, j): # 返回的是以i, j为终点的path的值
        if i == 0 and j == 0:
            return grid[0][0]
        up = float('inf')
        left = float('inf')
        if i - 1 >= 0:
            up = self.f1(grid, i - 1, j)
        if j - 1 >= 0:
            left = self.f1(grid, i, j - 1)
        return grid[i][j] + min(up, left)

    def f11(self, grid, i, j): # 返回的是以i，j为起点到终点n-1, m-1的值
        
        n,m = len(grid), len(grid[0])
        if i >= n-1 and j>=m-1:
            return 0
        down, right = float("inf"), float("inf") # TODO note 这里非常重要
        if i < n-1:
            down =  self.f11(grid, i+1, j)
        if j< m-1:
            right = self.f11(grid, i, j+1)
        
        return min(down, right) + grid[i][j] 
        # dp[i][j] = min(dp[i+1][j], dp[i][j+1]) + grid[i][j]
    
    def f21(self, grid, i, j):
        n,m = len(grid), len(grid[0])
        if i >= n-1 and j>=m-1:
            return 0
        if dp[i][j] != float("inf"):
            return dp[i][j]
        
        return min(dp[i+1][j] if i<n-1 else float('inf'), dp[i][j+1]) if j<m-1 else float('inf')+ grid[i][j])

    # 记忆化搜索
    def minPathSum2(self, grid):
        n = len(grid)
        m = len(grid[0])
        dp = [[-1] * m for _ in range(n)]
        return self.f2(grid, n - 1, m - 1, dp)

    def f2(self, grid, i, j, dp):
        if dp[i][j] != -1:
            return dp[i][j]
        ans = float('inf')
        if i == 0 and j == 0:
            ans = grid[0][0]
        else:
            up = float('inf')
            left = float('inf')
            if i - 1 >= 0:
                up = self.f2(grid, i - 1, j, dp)
            if j - 1 >= 0:
                left = self.f2(grid, i, j - 1, dp)
            ans = grid[i][j] + min(up, left)
        dp[i][j] = ans
        return ans

    # 严格位置依赖的动态规划
    def minPathSum3(self, grid):
        n = len(grid)
        m = len(grid[0])
        dp = [[0] * m for _ in range(n)]
        dp[0][0] = grid[0][0]
        for i in range(1, n):
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        for j in range(1, m):
            dp[0][j] = dp[0][j - 1] + grid[0][j]
        for i in range(1, n):
            for j in range(1, m):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        return dp[n - 1][m - 1]

    # 严格位置依赖的动态规划 + 空间压缩技巧
    def minPathSum4(self, grid):
        n = len(grid)
        m = len(grid[0])
        dp = [0] * m
        dp[0] = grid[0][0]
        for j in range(1, m):
            dp[j] = dp[j - 1] + grid[0][j]
        for i in range(1, n):
            dp[0] = dp[0] + grid[i][0]
            for j in range(1, m):
                dp[j] = min(dp[j - 1], dp[j]) + grid[i][j]
        return dp[m - 1]


    def f4(grid):
        dp = [0] * len(grid[0])
        n, m = len(grid), len(grid[0])

        for i in range(n):
            for j in range(m):
                dp[j] = min(dp[j], dp[j-1] if j>=1 else float("inf")) + grid[i][j]
        return dp[m-1]
    # for $ \any a ∈ b$