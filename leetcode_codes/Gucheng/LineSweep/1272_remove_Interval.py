class Solution:
    def removeInterval1(self, intervals: List[List[int]], toBeRemoved: List[int]) -> List[List[int]]:
        # 这个也对
        res = []
        rs, rt = toBeRemoved
        for (s,t) in intervals:
            # intersect
            s_in, t_in = max(s, rs), min(t, rt)
            if s_in >= t_in:
                res.append([s, t])
                continue
            # removed
            s1, t1 = s, max(s, s_in)
            s2, t2 = min(t_in, t), t

            if t1-s1>0:
                res.append([s1, t1])
            if t2-s2 >0:
                res.append([s2, t2])
        return res
    
    def removeInterval(self, intervals: List[List[int]], toBeRemoved: List[int]) -> List[List[int]]:
        res = []
        for i in intervals:
            # If no overlap, add the interval as is
            if i[1] < toBeRemoved[0] or i[0] > toBeRemoved[1]:
                res.append(i)
            else:
                # If there is some part of the interval before toBeRemoved, add that part
                if i[0] < toBeRemoved[0]:
                    res.append([i[0], toBeRemoved[0]])
                # If there is some part of the interval after toBeRemoved, add that part
                if i[1] > toBeRemoved[1]:
                    res.append([toBeRemoved[1], i[1]])
                # 如果两个都相等 就整个[s, t]被删除了
                # 注意相交有四种形态 s, rs, t, rt 或者 rs, s, rt, t 或者 rs, s, t, rt. s, rs, rt, t
        return res