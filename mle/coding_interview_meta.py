from typing import *
def merge_two_interval_1(l1, l2):

    if l1 == []: return l2
    if l2 == []: return l1
    
    p1, p2 = 0, 0
    s1, t1 = l1[0]
    s2, t2 = l2[0]
    res = []
    if s1 <= s2: #TODO correction: s1<s2
        res.append(l1[0])
    else:
        res.append(l2[0])


    while p1 < len(l1) or p2 < len(l2): # todo : or

        rs, rt = res[-1]
        if p1 == len(l1):
            _s, _t = l2[p2]
            p2 += 1
        elif p2 == len(l2):
            _s, _t = l1[p1]
            p1 += 1
        # check if they are overlap
        elif l1[p1][0] <= l2[p2][0]:
            _s, _t = l1[p1]
            p1 += 1
        else:
            _s, _t = l2[p2]
            p2 += 1
        
        # check if [s1, t1] intersect with [rs, rt]
        if _s <= rt: # TODO: originally it is '_s > rt or rs > _t '
            res[-1] = [min(_s, rs), max(_t, rt)] # typo in the interview: t1-> _t
        else:
            res.append([_s, _t])

    return res

def merge_two_interval_2(l1, l2): 
    # edge case : 

    # first merge l1[0] and l2[0] put it in the res
    # each we take the res[-1] and use it to merge intervals in l1 or l2
    p1, p2 = 0, 0
    res = []

    if l1 == []: return l2
    if l2 == []: return l1
    s1, t1 = l1[0]
    s2, t2 = l2[0]

    if s1 <= s2: # TODO originally written as s1 < s2 which is incorrect
        res.append([s1, t1])
    else:
        res.append([s2, t2]) # TODO originally missing else

    while p1 < len(l1) and p2 < len(l2): # TODO : or

        s1, t1 = l1[p1]
        s2, t2 = l2[p2]
        rs, rt = res[-1]
        # check if they are overlap
        if s1 <= s2:
            _s, _t = s1, t1
            p1 += 1
        else:
            _s, _t = s2, t2
            p2 += 1
            # check if [s1, t1] intersect with [rs, rt]
        if rt >= _s: # TODO as I can remember I wrote an incorrect condition without NOT and which is just a condition of 'non overlap'
            res[-1] = [min(_s, rs), max(_t, rt)] # TODO as I can remember I have a typo in the interview: t1-> _t
        else:
            res.append([_s, _t])
    """
    extension part (I thought about this part for the first few sections to handle the rest of the list but probablly got erased in a mess)
    Use 'or' in the above while loop conditon to handle when p1 == len(l1) or p2 == len(l2) should be better
    """
    if p1 == len(l1): 
        p1 = p2
        l1 = l2
    while p1 < len(l1):
        s1, t1 = l1[p1]
        rs, rt = res[-1]
        if rt >= s1:
            res[-1] = [min(_s, rs), max(_t, rt)]
        else:
            res.append([s1, t1])
        p1 += 1

    return res




l1, l2 = [[1,2], [3,4], [5,6]], [[10, 11], [12, 13]]
# l1, l2 = [[1,2], [3,4], [5,1000], [1200,1300]], [[-1,1], [10, 11], [12, 13]]
# l1, l2 = [[1,2], [3,4], [5,1000], [1200,1300]], [[-1,6], [10, 11], [12, 13]]
# l1, l2 = [], [[1,2], [5,6], [9, 10]]
# l1, l2 = [[10, 11], [12, 13]], [[1,2], [3,4], [5,1000]]

print(merge_two_interval_1(l1, l2))
print(merge_two_interval_2(l1, l2))