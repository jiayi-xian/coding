
from typing import List
import copy
def shortestPathAllKeys(grid: List[str]) -> int:

    # components: 
    # walkable ways at [x, y]
    # 
    keys = []


    # 对BFS来说 我们建立 visited set ( "|".join([x,y,"".join(sorted(keys)))), q, 记录步数
    G = [list(item) for item in grid]
    nrow, ncol = len(G), len(G[0])
    neis = lambda x,y : [[nx, ny] for nx, ny in [[x-1, y],[x, y-1],[x, y+1],[x+1, y]] if 0<= nx <= nrow-1 and  0<= ny <= ncol-1]

    for idx_r, rows in enumerate(G):
        for idx_c, item in enumerate(rows):
            if item == "@":
                start = [idx_r, idx_c]
            if item.islower():
                keys.append([idx_r, idx_c, item])
    import string
    step = 0
    cur_keys = [0 for _ in range(26)]
    open_locks = [0 for _ in range(26)]
    keys_get = 0
    visited = set()
    if len(keys) == 0:
        return 0
    from collections import deque
    def zip_state(x, y, cur_keys, keys_get):
        keys_str = ",".join(list(map(str, cur_keys)))
        #locks_str = ",".join(locks)
        return "|".join([str(x),str(y),keys_str, str(keys_get)])
    
    def unzip_state(state):
        x,y,keys_str, keys_get_str = state.split("|")
        cur_keys = keys_str.split(",")
        cur_keys = list(map(int, cur_keys))
        keys_get = int(keys_get_str)
        return int(x), int(y), cur_keys, keys_get
    
    def get_path(prenodes, state):
        path = [state]
        cnt = 0
        while state is not None:
            cnt += 1
            state = prenodes[state]
            path.append(state)
        return path[::-1]
            

    visited=set([zip_state(start[0], start[1], cur_keys, keys_get)])
    q= deque([zip_state(start[0], start[1], cur_keys, keys_get)]) # x,y,cur_keys,
    prenodes = {}
    prenodes[zip_state(start[0], start[1], cur_keys, keys_get)] = None
    path = []
    while q:
        L = len(q)
        for _ in range(L):
            cs = q.popleft()
            x, y, cur_keys, keys_get = unzip_state(cs)
            if keys_get == len(keys): 
                path = [cs]
                while prenodes[cs] is not None:
                    path.append(cs)
                    cs = prenodes[cs]
                return step, path[::-1]
            for nx, ny in neis(x,y):
                nc = G[nx][ny]
                ns0 = zip_state(nx, ny, cur_keys, keys_get)
                if nc == "." or nc == "@":
                    if ns0 not in visited:
                        ns = ns0
                        q.append(ns)
                        prenodes[ns] = cs
                        visited.add(ns)
                elif nc.islower(): # get a keys
                    if cur_keys[ord(nc)-ord("a")]==0:
                        cur_keys_ = copy.deepcopy(cur_keys)
                        cur_keys_[ord(nc)-ord("a")] += 1
                        # '4|2|1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0|5'
                        ns = zip_state(nx, ny, cur_keys_, keys_get+1)
                        if keys_get + 1 == len(keys): 
                            prenodes[ns] = cs
                            return step+1, get_path(prenodes,ns)
                    else:
                        ns = ns0
                    if ns not in visited:
                        q.append(ns)
                        prenodes[ns] = cs
                        visited.add(ns)
                elif nc.isupper() and cur_keys[ord(nc.lower())-ord("a")]>0: # encounter a lock and have a key  
                    if ns0 not in visited:
                        #cur_keys[ord(nc.lower())-ord("a")] -=1
                        #G[nx][ny] = "."
                        ns = ns0
                        q.append(ns)
                        prenodes[ns] = cs
                        visited.add(ns)
                else:
                    continue
        step += 1
    return visited

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
