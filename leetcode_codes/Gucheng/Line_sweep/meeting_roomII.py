class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int: # O(2n*log(2n)) 扫描线
        
        times = []
        c = 0
        maxC = 0
        for s, t in intervals:
            times.append((s, 1))
            times.append((t, -1))
            
        times.sort()
        
        for time, sig in times:
            c += sig
            
            maxC = max(maxC, c)
            
        
        return maxC
import heapq
    def minMeetingRooms(self, intervals: List[List[int]]) -> int: # n^2logn

        # if two schedules overlap, we need to add 1 more room
        # for si, we can release room for any schedule tj <= si we need to maintain a heap to sort tj that have been seen

        time = sorted(intervals)
        q = []
        heapq.heapify(q) # -1
        maxR = 0

        for s, t in time:
            while q and q[0]<=s:
                heapq.heappop(q)
            heapq.heappush(q, t)
            maxR = max(maxR, len(q))

        return maxR