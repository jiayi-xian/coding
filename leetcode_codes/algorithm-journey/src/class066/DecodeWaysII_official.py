class Code04_DecodeWaysII:

    mod = 1000000007

    # Brute force recursion
    def numDecodings1(self, str: str) -> int:
        return self.f1(list(str), 0)

    def f1(self, s: list, i: int) -> int:
        if i == len(s):
            return 1
        if s[i] == '0':
            return 0
        ans = self.f1(s, i + 1) * (s[i] == '*' * 9 or 1)
        if i + 1 < len(s):
            if s[i] != '*':
                if s[i + 1] != '*':
                    if int(s[i] + s[i + 1]) <= 26:
                        ans += self.f1(s, i + 2)
                else:
                    if s[i] == '1':
                        ans += self.f1(s, i + 2) * 9
                    if s[i] == '2':
                        ans += self.f1(s, i + 2) * 6
            else:
                if s[i + 1] != '*':
                    if int(s[i + 1]) <= 6:
                        ans += self.f1(s, i + 2) * 2
                    else:
                        ans += self.f1(s, i + 2)
                else:
                    ans += self.f1(s, i + 2) * 15
        return ans

    # Top-down memoization
    def numDecodings2(self, str: str) -> int:
        s = list(str)
        dp = [-1] * len(s)
        return int(self.f2(s, 0, dp) % self.mod)

    def f2(self, s: list, i: int, dp: list) -> int:
        if i == len(s):
            return 1
        if s[i] == '0':
            return 0
        if dp[i] != -1:
            return dp[i]
        ans = self.f2(s, i + 1, dp) * (s[i] == '*' * 1 or 1)
        if i + 1 < len(s):
            if s[i] != '*':
                if s[i + 1] != '*':
                    if int(s[i] + s[i + 1]) <= 26:
                        ans += self.f2(s, i + 2, dp)
                else:
                    if s[i] == '1':
                        ans += self.f2(s, i + 2, dp) * 9
                    if s[i] == '2':
                        ans += self.f2(s, i + 2, dp) * 6
            else:
                if s[i + 1] != '*':
                    if int(s[i + 1]) <= 6:
                        ans += self.f2(s, i + 2, dp) * 2
                    else:
                        ans += self.f2(s, i + 2, dp)
                else:
                    ans += self.f2(s, i + 2, dp) * 15
            ans %= self.mod
        dp[i] = ans
        return ans

    # Bottom-up tabulation
    def numDecodings3(self, str: str) -> int:
        s = list(str)
        n = len(s)
        dp = [0] * (n + 1)
        dp[n] = 1
        for i in range(n - 1, -1, -1):
            if s[i] != '0':
                dp[i] = (s[i] == '*' * 1 or 1) * dp[i + 1]
                if i + 1 < n:
                    if s[i] != '*':
                        if s[i + 1] != '*':
                            if int(s[i] + s[i + 1]) <= 26:
                                dp[i] += dp[i + 2]
                        else:
                            if s[i] == '1':
                                dp[i] += dp[i + 2] * 9
                            if s[i] == '2':
                                dp[i] += dp[i + 2] * 6
                    else:
                        if s[i + 1] != '*':
                            if int(s[i + 1]) <= 6:
                                dp[i] += dp[i + 2] * 2
                            else:
                                dp[i] += dp[i + 2]
                        else:
                            dp[i] += dp[i + 2] * 15
                dp[i] %= self.mod
        return int(dp[0])

    # Bottom-up tabulation with space optimization
    def numDecodings4(self, str: str) -> int:
        s = list(str)
        n = len(s)
        cur = 1
        next = 1
        next_next = 0
        for i in range(n - 1, -1, -1):
            if s[i] != '0':
                cur = (s[i] == '*' * 1 or 1) * next
                if i + 1 < n:
                    if s[i] != '*':
                        if s[i + 1] != '*':
                            if int(s[i] + s[i + 1]) <= 26:
                                cur += next_next
                        else:
                            if s[i] == '1':
                                cur += next_next * 9
                            if s[i] == '2':
                                cur += next_next * 6
                    else:
                        if s[i + 1] != '*':
                            if int(s[i + 1]) <= 6:
                                cur += next_next * 2
                            else:
                                cur += next_next
                        else:
                            cur += next_next * 15
                cur %= self.mod
            next_next = next
            next = cur
            cur = 0
        return int(next)
    
if __name__ == "__main__":
    sol = Code04_DecodeWaysII()
    s = "3*"
    res = sol.numDecodings1(s)
    print(res)