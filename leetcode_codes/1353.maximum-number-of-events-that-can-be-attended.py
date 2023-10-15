#
# @lc app=leetcode id=1353 lang=python3
#
# [1353] Maximum Number of Events That Can Be Attended
#
import heapq
from typing import *
# @lc code=start
class Solution:
    def maxEvents(self, events: List[List[int]]) -> int:
        
        minh = [] # [es, et]
        heapq.heapify(minh)
        cur, last = 0, 0
        count = 0
        
        for es, et in events:

            # compute how many events one can attend during [last, es)
            while len(minh) != 0 and minh[0][0] <= last <= minh[0][1] and last < es:
                heapq.heappop(minh)
                last += 1
                count += 1
            
            # pop events expired before es
            while len(minh) != 0 and minh[0][1] < es:
                heapq.heappop(minh)
            
            last = es
            
            # push current events into minh
            heapq.heappush(minh, [es, et])
            
        
        return count


# @lc code=end

