from collections import Counter
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

def minWindow(self, s: str, t: str) -> str:

    res, l = float("inf"), 0
    cnt = {}
    pattern = {c:True for c in t}

    for i in range(len(s)):
        c = s[i]
        if c in pattern:
            if c not in cnt:
                cnt[c] = 0
            cnt[c] += 1
        while len(cnt) == len(pattern):
            lc = s[l]
            if lc in cnt:
                cnt[lc] -= 1
                if cnt[lc] == 0:
                    del cnt[lc]
            l += 1
        
        res = min(res, i-l)
        print(pattern)
    return res