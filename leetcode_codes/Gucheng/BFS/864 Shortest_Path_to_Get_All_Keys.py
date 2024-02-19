
from typing import *
from collections import *
from itertools import *
import copy
class Solution:
    def shortestPathAllKeys(self, grid: List[str]) -> int:

        # 在最短路径上，不可能我们经过了某个房间两次，并且这两次我们拥有钥匙的情况是完全一致的。
        # states: 
        G = grid
        m, n = len(grid), len(grid[0])


        #  search for start
        # start = [G[x][y] for x in range(m) for y in range(n) if G[x][y] =="@"]
        start = [[x,y] for x in range(m) for y in range(n) if G[x][y] =="@"]
        # search for number of keys
        nkey = sum([1 for x in range(m) for y in range(n) if G[x][y].islower()])
        dirs = (-1, 0, 1, 0, -1)
        neis = lambda x, y : [ [x+dx, y+dy] for dx, dy in pairwise(dirs) if 0<=x+dx < m and 0 <= y+dy < n]

        start_node = (start[0][0], start[0][1], 0)
        q = [(start[0][0], start[0][1], 0)]
        q = deque(q)
        vis = set(start_node)
        steps = 0

        while q:
            for _ in range(len(q)):

                x, y, ks = q. popleft()
                if ks == (1<<nkey) - 1: return steps

                for nx, ny in neis(x, y):
                    cur = G[nx][ny]

                    if cur.islower(): # key
                        ks_new = ks | 1 << (ord(cur)-ord("a"))
                        if (nx, ny, ks_new) not in vis:
                            vis.add((nx, ny, ks_new))
                            q.append((nx, ny, ks_new))
                    elif cur.isupper() and (ks & (1 << (ord(cur) - ord("A")))): # lock with key applicable
                        if (nx, ny, ks) not in vis:
                            vis.add((nx, ny, ks))
                            q.append((nx,ny,ks))
                    elif cur in ("@", "."): # empty 要注意@的情况 这里不能是cur != "#" 前面的情况都是允许step on的情况，而cur != "#"不满足这个条件，这包括了碰到锁单没有钥匙的情况
                        if (nx, ny, ks) not in vis:
                            vis.add((nx, ny, ks))
                            q.append((nx,ny,ks))
            steps += 1

        return -1

grids =[
["Dd#b@",
 ".fE.e",
 "##.B.",
 "#.cA.",
 "aF.#C"]
]

for grid in grids:
    print(shortestPathAllKeys(grid))

x = ["@...a",".###A","b.BCc"]


def check_kth_bit(state, k):
    mask = 1 << k  # 将 1 向左移动 k 位，创建掩码
    if state & mask:
        return True  # 第 k 位为 1
    else:
        return False  # 第 k 位为 0