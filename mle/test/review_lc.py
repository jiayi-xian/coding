

class Dsu:

    def __init__(self, N) -> None:
        self.p = list(range(N))

    def find_parent(self, v):
        if self.p[v] !=v:
            self.p[v] = self.find_parent(self.p[v])
    
    def union(self, u, v):
        self.find_parent(u)
        self.find_parent(v)
        self.p[u] = self.p[v]


class Solution:
    def findOrder(self, numgraphs: int, edges: List[List[int]]) -> List[int]:
        
        
        def dfs(pre):
            
            if visited[pre] == 2:
                return False # isCircle
            
            visited[pre] = 2 # visiting
            
            if pre in graph:
                for nex in graph[pre]:
                    if visited[nex] != 1:
                        res = dfs(nex)
                        if res == False:
                            return False
                
            
            stack.append(pre)
            visited[pre] = 1
            
            return True
                
        
        
        graph = {}
        num = 0
        visited = [0 for _ in range(numgraphs)]
        
        for nex, pre in edges:
            if pre not in graph:
                graph[pre] = []
            graph[pre].append(nex)
                
                
        stack = []
        
        for u in range(numgraphs):
            if visited[u] == 0:
                if dfs(u) == False:
                    return []
                
        return stack[::-1] if len(edges)>0 else list(range(numgraphs))


class Solution:

    def inorderTraversal(self, node):

        cur = node
        res = []
        stack = []
        
        while stack or cur:
            while cur:
                stack.append(cur)
                cur = cur.left
            
            cur = stack.pop()
            res.append(cur.val)
            cur = cur.right
        
        return res


class Djistra