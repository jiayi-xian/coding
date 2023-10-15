from collections import deque, defaultdict
from typing import List
def minJumps_tle(arr: List[int]) -> int:

    q = deque()
    visited = set()
    step = 0
    q.append(0)
    visited.add(0)
    val2idx = defaultdict(set)
    for idx, val in enumerate(arr):
        val2idx[val].add(idx)

    val2idx[arr[0]].remove(0)
    
    while q:
        L = len(q)
        for _ in range(L):
            cur = q.popleft()
            val = arr[cur]
            if cur == len(arr)-1:
                return step
            neighbors = val2idx[val].copy()
            for nex in neighbors:
                if nex != cur and nex not in visited:
                    q.append(nex)
                    visited.add(nex)
                    val2idx[val].remove(nex)
            if cur + 1 <len(arr) and (cur+1 not in visited):
                q.append(cur+1)
                visited.add(cur+1)
            if cur - 1 >=0 and (cur-1 not in visited):
                q.append(cur-1)
                visited.add(cur-1)
        step += 1 #todo 注意位置
            
def minJumps(arr: List[int]) -> int:
    
    n = len(arr)
    q1, q2, q3 = {0}, {n-1}, set()
    visited = set()
    step = 0
    visited.add(0)
    visited.add(n-1) # TODO how to add/remove multiple elements in set

    val2idx = defaultdict(set)
    for idx, val in enumerate(arr):
        val2idx[val].add(idx)

    val2idx[arr[0]].discard(0) # 可能只有len 1
    #val2idx[arr[n-1]].discard(n-1) # 不能去掉 1.若idx1与最后一个元素相同，一开始去掉了最后一个元素的idx会导致idx为1时reach不到last index 导致返回错误的结果 2.可能只有len 1 被去掉0就剩空集 remove会返回错误
    
    while q1: #TODO q1 or q2
        L = len(q1)
        for _ in range(L):
            cur = q1.pop()
            val = arr[cur]
            if cur in q2:
                return step
            
            for nex in [cur-1, cur+1]+ list(val2idx[val]):
                if nex in q2: return step + 1
                if 0<= nex <=n-1 and nex!= cur and nex not in visited:
                    q3.add(nex)
                    visited.add(nex)
                if nex in val2idx[val]:
                    val2idx[arr[nex]].discard(nex) # 注意不要写成val2[idx][val]
        
        q1, q3 = q3, set()
        step += 1 #todo 注意位置
        if len(q1) > len(q2): q1, q2 = q2, q1
    return -1