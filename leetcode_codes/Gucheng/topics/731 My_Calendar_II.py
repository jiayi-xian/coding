from collections import defaultdict
class MyCalendarTwo:

    # segment tree with lazy propagation for range max query
    # data structure: defaultdict. key: index (to locate children. e.g. i -> 2i, 2i + 1), value: range max
    # idea: each booking can be considered as + 1 booking for every time point in the range 
    # Time: O(logd), where d is the depth of segment tree
    # Space: O(n), where n is the number of bookings
    
    # [LOWER, UPPER)
    LOWER = 0
    UPPER = 1e9 + 1
    
    def __init__(self):
        self.tree = defaultdict(int)
        self.lazy = defaultdict(int)

    def book(self, start: int, end: int) -> bool:
		# if the max booking in [start, end) is 2
		# the booking is not allowed
        if self.query(start, end) == 2:
            return False
		# else, update interval [start, end) by +1
        self.update(start, end)
        return True
    
    def query(self, start, end, l=LOWER, r=UPPER, index=1):
        """
		node information: l, r, index
        query interval: start, end
		"""
		
        if start == l and end == r:
            return self.tree[index]

        m = (l + r) // 2

        if end <= m:
            return self.lazy[index] + self.query(start, end, l, m, 2 * index)
        elif m <= start:
            return self.lazy[index] + self.query(start, end, m, r, 2 * index + 1)
        else: # start < m < end
            return self.lazy[index] + max(
                self.query(start, m, l, m, 2* index),
                self.query(m, end, m, r, 2 * index + 1)
            )
        
    def update(self, start, end, l=LOWER, r=UPPER, index=1):
        """
		node information: l, r, index
        query interval: start, end
		"""
		
        if start == l and end == r:
            self.tree[index] += 1
            self.lazy[index] += 1
            return

        m = (l + r) // 2

        if end <= m:
            self.update(start, end, l, m, 2 * index)
        elif m <= start:
            self.update(start, end, m, r, 2 * index + 1)
        else: # start < m < end
            self.update(start, m, l, m, 2 * index)
            self.update(m, end, m, r, 2 * index + 1)

        self.tree[index] = self.lazy[index] + max(self.tree[2 * index], self.tree[2 * index + 1])
        return

# Your MyCalendarTwo object will be instantiated and called as such:
# obj = MyCalendarTwo()
# param_1 = obj.book(start,end)
obj = MyCalendarTwo()
intervals = [[10,20],[50,60],[10,40],[5,15],[5,10],[25,55]]
for s, t in intervals:
    parma_1 = obj.book(s, t)
