
def findall_in_string(txt = "test test test test", pattern = "test"):
    res = []
    i = 0
    while i < len(txt):
        idx = txt.find(pattern, i)
        if idx != -1: # unfound, find will return -1
            res.append(idx)
            i = idx + len(pattern)
        else:
            i += 1
    print(res)
    return res