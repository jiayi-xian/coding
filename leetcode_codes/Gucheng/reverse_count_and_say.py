from collections import defaultdict
def reverseCountSay(s):
    memo = defaultdict(list)

    def recursive(s):
        if len(s) == 0:
            return [[]]
        elif s in memo:
            return memo[s]
        for i in range(1, len(s)):
            curNum, count = s[i], int(s[:i])
            for subRes in recursive(s[i + 1:]):
                memo[s].append([curNum * count] + subRes)
        return memo[s]
    recursive(s)
    return [''.join(output) for output in memo[s]]

print(reverseCountSay("222"))


"""
count_and_say(211) -> 1221
reverse(1221) -> 211
reverse(12211) -> 21111...11, 12*"2"+"1", 
"""

