import networkx as nx
import random
import numpy as np

def add_edges_rand(G, p=0.5):
    """
    
    Parameters:
    -----------
    G: graph. dict of dict
    Returns:
    --------
    
    """
    for u in G:
        #for v in G[u]:
        for v in G:
            if u!=v:
                r = random.uniform(0,1)
                if r < p:
                    G[u][v] = 1
    
    return G


def pagerank(G, alpha, W, nstart, p, max_iter=100, tol=1.0e-6):
    """
    
    https://www.geeksforgeeks.org/page-rank-algorithm-implementation/
    return pagerank of a graph
    Parameters:
    -----------
    G: graph, networkx object, dict of dict of dict
    W: dict of dict: weight on edge
    alpha: damping factor
    p: dict. personalization vector for each node.
    nstart: dict. start pagerank vector for each node, else uniform
    Returns:
    --------
    x: dict: key: node; val: pagerank.
    """ 
    N = G.number_of_nodes() # warning: len(G) != # of nodes
    if nstart is None:
        x = dict.fromkeys(G, 1.0/N)
    else:
        # normalized nstart vector
        s = float(sum(nstart.values()))
        x = dict((k, v/s) for k, v in nstart.items())

    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        for u in x:
            for v in W[u]:
                x[v] += alpha* xlast[u] * W[u][v]
            x[u] += (1-alpha)*p[u]

        err = sum(abs(xlast[u] - x[u]) for u in x)
        if err < N*tol:
            return x
    
    if err >= N*tol:
        return False



def randomWalk(G,  max_iter = 100, p=None, nstart=None):
    """
    https://www.geeksforgeeks.org/implementation-of-page-rank-using-random-walk-method-in-python/
    Parameters:
    -----------
    G: dict of dict: node: neighboring nodes: weight
    p: dict. personalization vector
    max_iter: maximal iteration allowed
    Returns:
    --------
    x: dict: pagerank for each node
    """
    N = G.number_of_nodes()
    if p is None:
        p = dict.fromkeys(G, 1)
    else:
        assert len(G) == len(p)
        s = float(sum(p.values()))
        p = dict((k, v / s) for k, v in p.items())

    
    if nstart is None:
        x = dict.fromkeys(G, 1) # ???
    else:
        assert len(G) == len(nstart)
        s = sum(nstart[u] for u in nstart.keys())
        #x = {k:v/s for k, v in nstart}
        x = dict((k, v/s) for k,v in nstart.items())
    
    # try debug
    cur = np.random.choice(N, p=p)
    x[cur] += 1
    nodes = np.array([[k,v] for k, v in p])
    for _ in range(max_iter):
        if G[cur]:
            neighbors = np.array([[k,v] for k,v in G[cur].items()])
            nex = np.random.choice(neighbors[:,0], p = neighbors[:,1])
        else:
            nex = np.random.choice(N, p=p)
        x[nex] += 1
        cur = nex
    return x

def nodes_sorted(g, points):
	t = np.array(points)
	t = np.argsort(-t)
	return t
        
def check1():
    # Main
    # 1. Create a directed graph with N nodes
    g = nx.DiGraph()
    N = 5
    g.add_nodes_from(range(N))

    # 2. Add directed edges in graph
    g = add_edges_rand(g, 0.4)

    # 3. perform a random walk
    points = randomWalk(g)

    # 4. Get nodes rank according to their random walk points
    sorted_by_points = nodes_sorted(g, points)
    print("PageRank using Random Walk Method")
    print(sorted_by_points)

    # p_dict is dictionary of tuples
    p_dict = nx.pagerank(g)
    p_sort = sorted(p_dict.items(), key=lambda x: x[1], reverse=True)

    print("PageRank using inbuilt pagerank method")
    for i in p_sort:
        print(i[0], end=", ")



'''
discuss here
dp: dp[i][j] : if text[:i+1] matches pattern[:j+1]
dp[0][0] = 1 ;

text: 
x x x a
pattern
x x x x Y *  
dp[i][j]: true if s[:i] matches p[:j] else False

s, p = text, pattern
m, n = len(p), len(s)

for i in range(n):
  dp[i][0] = False

if p[0] != "*":
  for j in range(m):
    dp[0][j] = False
else:
  for j in range(m):
    dp[0][j] = True
 
dp[0][0] = True


for i in range(1, n):
  for j in range(1, m):
    if p[j] == s[i]:
      dp[i+1][j+1] |= dp[i][j]
    elif p[j] == '*':
      dp[i+1][j+1] |= dp[i+1][j-1] or (dp[i][j+1] if p[j-1] == s[i-1])
    elif p[j] == ".":
      dp[i][j+1] |= dp[i-1][j]

return dp[n+1][n+1]
  

corner case: "*a"
'''

'''
my_string = "hello" => "olleh"

my_string[2] = "L" # wont work, strings are immutable

'''

def is_match(text, pattern):
  #dp[i][j]: true if s[:i] matches p[:j] else False

  s, p = text, pattern
  m, n = len(p), len(s)
  
  dp = [[False for _ in range(m)] for _ in range(n)]


  if p[0] == "*":
    for j in range(m):
      dp[0][j] = True

  dp[0][0] = True


  for i in range(1, n):
    for j in range(1, m):
      if p[j] == s[i]:
        dp[i+1][j+1] |= dp[i][j]
      elif p[j] == '*':
        dp[i+1][j+1] |= dp[i+1][j-1]
        dp[i+1][j+1] |= dp[i][j+1] if p[j-1] == s[i-1] else dp[i+1][j+1]
      elif p[j] == ".":
        dp[i][j+1] |= dp[i-1][j]

  return dp[n+1][n+1]
