from collections import defaultdict
 
class Graph():
    def __init__(self,vertices):
        self.graph = defaultdict(list)
        self.V = vertices
 
    def addEdge(self,u,v):
        self.graph[u].append(v)
    
    # dfs + coloring

    def isCycle1(self, u, visited, graph):
        """
        Check if any path starting from u forms a cycle
        Parameters:
        -----------
        graph: dict {u:[v1, v2, ...]} (u,vi) is an edge in graph
        u: node
        visited: (n,)
        list of int: 1: visited; 0: unvisited; 2: visiting
        Returns:
        bool, True if there is a cycle else False
        --------
        """
        # base case: a cycle is detected
        if visited[u] == 2:
            return True
        
        for v in graph[u]:
            # iterate over all the neigbours of u to check if there is a cycle
            if visited[u] != 1: # check unvisited nodes
                visited[u] = 2 # marked as browsing
                if self.isCycle1(v, visited, graph):
                    return True
        
        visited[u] = 1
        return False
    
    def isCyclic1(self):
        visited = [False] * (self.V + 1)
        for node in range(self.V):
            if visited[node] != 1:
                if self.isCycle1(node,visited,self.graph) == True:
                    return True
        return False

        # topological sort: topological sort is implemented using dfs, which is very similar to the above
        # The difference is instead of marking the nodes, storing nodes in stack by visiting order.

    def topoSort(self, u, stack, visited, graph):
        """
        return topographical sort ordering of nodes in a graph
        Parameters:
        -----------
        graph: dict {u:[v1, v2, ...]} (u,vi) is an edge in graph
        u: node
        visited: set of visited nodes.
        stack:
        list of nodes.
        Returns:
        stack
        --------
        """
        visited.add(u)
        for v in graph[u]:
            # iterate over all the neigbours of u to check if there is a cycle
            if v not in visited: # check unvisited nodes
                stack, visited = self.topoSort(v, visited, stack, graph)
        
        stack.append(u)
        return stack, visited
        # the way that checking if a cycle exists through toposort is to check the order sequence for each edge u->v in the topoSort stack. If for some edge u -> v, ord(u) > ord(v). Then there is a cycle 

    def isCycle2(self, v, visited, recStack):
 
        # Mark current node as visited and
        # adds to recursion stack
        visited[v] = True
        recStack[v] = True
 
        # Recur for all neighbours
        # if any neighbour is visited and in
        # recStack then graph is cyclic
        for neighbour in self.graph[v]:
            if visited[neighbour] == False:
                if self.isCycle2(neighbour, visited, recStack) == True:
                    return True
            elif recStack[neighbour] == True:
                return True
 
        # The node needs to be popped from
        # recursion stack before function ends
        recStack[v] = False
        return False
 
    # Returns true if graph is cyclic else false
    def isCyclic2(self):
        visited = [False] * (self.V + 1)
        recStack = [False] * (self.V + 1)
        for node in range(self.V):
            if visited[node] == False:
                if self.isCycle2(node,visited,recStack) == True:
                    return True
        return False
 
g = Graph(4)
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)
if g.isCyclic() == 1:
    print("Graph contains cycle")
else:
    print("Graph doesn't contain cycle")