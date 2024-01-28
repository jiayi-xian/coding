from collections import OrderedDict, Counter, defaultdict
import copy
# input: string s, list of string: stickers

def spell(target, stickers):

    cnt_ss = [Counter(sti) for sti in stickers]
    cnt_t = OrderedDict(sorted([(key, val) for key, val in Counter(target).items()])) #TODO .items()
    cnt_set = [set(sti) for sti in stickers]

    memo = {}
    memo[""] = 0
    
    
    minx = float("inf")
    def helper(cnt_t, steps):
        # recursively find sticker to spell out target 
        # return the minimum num of stickers to spell out the current target (cnt_t)
        nonlocal minx
        if len(cnt_t) == 0: # done
            minx = min(minx, steps)
            return 0
        
        token = "".join([key+str(val) for key, val in cnt_t.items()]) # #.items()
        if token in memo:
            return memo[token]
        
        # try any sticker to reduce target
        min_step = float("inf")

        for cnt_s in cnt_ss:
            if token[0] in cnt_s: # always to choose the first character to remove (reduce recursive branches)
                picks = set(cnt_s).intersection(set(cnt_t.keys()))
                cnt_ttemp = copy.copy(cnt_t)
                # spell with sticker cnt_s
                for c in picks:
                    cnt_ttemp[c] -= cnt_s[c]
                    if cnt_ttemp[c] <= 0:
                        del cnt_ttemp[c]
                # recursion, res: minimum num of stickers to spell out the cnt_ttemp (reduced target)
                res = 1 + helper(cnt_ttemp, steps+1)
                min_step = min(res, min_step)
        memo[token] = min_step
        return min_step

    token0 =  "".join([key+str(val) for key, val in cnt_t.items()]) #.items()
    print(minx)
    helper(cnt_t, 0)
    return memo[token0]

target = "swime"
stickers = ["sw","ice", "me"]
res = spell(target, stickers)
print(res)