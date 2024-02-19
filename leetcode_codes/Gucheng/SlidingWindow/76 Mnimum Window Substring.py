from collections import Counter

"""
1. 对于每个s[i] 如果属于t中的字符 则需求-1
2. 当满足需求时 -> needstrcnt = len(t), needstrcnt == 0
2.1 缩小窗口
left += 1 直至 needstr[lc] == 0
符合条件 记录长度

进入下一次iteration

"""


def minWindow(self, s: str, t: str) -> str:
    if len(s) < len(t):
        return ""
    needstr = collections.defaultdict(int)
    for ch in t:
        needstr[ch] += 1
    needcnt = len(t)
    res = (0, float('inf'))
    start = 0
    for end, ch in enumerate(s):
        if needstr[ch] > 0:
            needcnt -= 1
        needstr[ch] -= 1
        if needcnt == 0:
            while True:
                tmp = s[start]
                if needstr[tmp] == 0:
                    break
                needstr[tmp] += 1
                start += 1
            if end - start < res[1] - res[0]:
                res = (start, end)
            needstr[s[start]] += 1
            needcnt += 1
            start += 1
    return '' if res[1] > len(s) else s[res[0]:res[1]+1]

def solution(s, t):
    res, l = float("inf"), 0
    cnt = {}
    win = ""
    pattern = Counter(t)

    def isContain(cnt, pattern):
        for key in pattern.keys():
            if key not in cnt:
                return False
            if cnt[key] - pattern[key] < 0:
                return False
        return True

    for i in range(len(s)):
        c = s[i]
        if c in pattern:
            if c not in cnt:
                cnt[c] = 0
            cnt[c] += 1
        while isContain(cnt, pattern):
            if i-l+1 < res:
                win = s[l:i+1]
                res = min(res, i-l+1)

            lc = s[l]
            if lc in cnt:
                cnt[lc] -= 1
                if cnt[lc] == 0:
                    del cnt[lc]
            l += 1

    return win

s = "ADOBECODEBANC"
t = "ABC"
print(solution(s,  t))

#判断语句可以写得更加巧妙
"""
need = len(pattern)

if cnt[c] == pattern[c]:
    have += 1

while have == need:
    blabla

    if lc in cnt and cnt[lc] < pattern[lc]:
        have -= 1
    
    l += 1
"""