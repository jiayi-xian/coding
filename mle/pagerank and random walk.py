import networkx as nx
import random
import numpy as np

# Add directed edges in graph
def add_edges(g, pr):
	for each in g.nodes():
		for each1 in g.nodes():
			if (each != each1):
				ra = random.random()
				if (ra < pr):
					g.add_edge(each, each1)
				else:
					continue
	return g

# Sort the nodes
def nodes_sorted(g, points):
	t = np.array(points)
	t = np.argsort(-t)
	return t

# Distribute points randomly in a graph
def random_Walk(g):
	# https://www.geeksforgeeks.org/implementation-of-page-rank-using-random-walk-method-in-python/
	rwp = [0 for i in range(g.number_of_nodes())]
	nodes = list(g.nodes())
	r = random.choice(nodes)
	rwp[r] += 1
	neigh = list(g.out_edges(r))
	z = 0
	
	while (z != 10000):
		if (len(neigh) == 0):
			focus = random.choice(nodes)
		else:
			r1 = random.choice(neigh)
			focus = r1[1] # edge:(start, end)
		rwp[focus] += 1
		neigh = list(g.out_edges(focus))
		z += 1
	return rwp
	# random.choice(arr, weight = ((1-alpha)*p[0], (1-alpha)*p[1],. , alpha/m, (1-alpha)*p[j], ..., alpha/m, )) personalization random walk

def pagerank(G, alpha=0.85, personalization=None,
			max_iter=100, tol=1.0e-6, nstart=None, weight='weight',
			dangling=None):
	"""Return the PageRank of the nodes in the graph.

	PageRank computes a ranking of the nodes in the graph G based on
	the structure of the incoming links. It was originally designed as
	an algorithm to rank web pages.

	Parameters
	----------
	G : graph
	A NetworkX graph. Undirected graphs will be converted to a directed
	graph with two directed edges for each undirected edge.

	alpha : float, optional
	Damping parameter for PageRank, default=0.85.

	personalization: dict, optional
	The "personalization vector" consisting of a dictionary with a
	key for every graph node and nonzero personalization value for each node.
	By default, a uniform distribution is used.

	max_iter : integer, optional
	Maximum number of iterations in power method eigenvalue solver.

	tol : float, optional
	Error tolerance used to check convergence in power method solver.

	nstart : dictionary, optional
	Starting value of PageRank iteration for each node.

	weight : key, optional
	Edge data key to use as weight. If None weights are set to 1.

	dangling: dict, optional
	The outedges to be assigned to any "dangling" nodes, i.e., nodes without
	any outedges. The dict key is the node the outedge points to and the dict
	value is the weight of that outedge. By default, dangling nodes are given
	outedges according to the personalization vector (uniform if not
	specified). This must be selected to result in an irreducible transition
	matrix (see notes under google_matrix). It may be common to have the
	dangling dict to be the same as the personalization dict.

	Returns
	-------
	pagerank : dictionary
	Dictionary of nodes with PageRank as value

	Notes
	-----
	The eigenvector calculation is done by the power iteration method
	and has no guarantee of convergence. The iteration will stop
	after max_iter iterations or an error tolerance of
	number_of_nodes(G)*tol has been reached.

	The PageRank algorithm was designed for directed graphs but this
	algorithm does not check if the input graph is directed and will
	execute on undirected graphs by converting each edge in the
	directed graph to two edges.

	
	"""
	if len(G) == 0:
		return {}

	if not G.is_directed():
		D = G.to_directed()
	else:
		D = G

	# Create a copy in (right) stochastic form
	W = nx.stochastic_graph(D, weight=weight)
	N = W.number_of_nodes()

	# Choose fixed starting vector if not given
	if nstart is None:
		x = dict.fromkeys(W, 1.0 / N)
	else:
		# Normalized nstart vector
		s = float(sum(nstart.values()))
		x = dict((k, v / s) for k, v in nstart.items())

	if personalization is None:

		# Assign uniform personalization vector if not given
		p = dict.fromkeys(W, 1.0 / N)
	else:
		missing = set(G) - set(personalization)
		if missing:
			raise NetworkXError('Personalization dictionary '
								'must have a value for every node. '
								'Missing nodes %s' % missing)
		s = float(sum(personalization.values()))
		p = dict((k, v / s) for k, v in personalization.items())

	if dangling is None:

		# Use personalization vector if dangling vector not specified
		dangling_weights = p
	else:
		missing = set(G) - set(dangling)
		if missing:
			raise NetworkXError('Dangling node dictionary '
								'must have a value for every node. '
								'Missing nodes %s' % missing)
		s = float(sum(dangling.values()))
		dangling_weights = dict((k, v/s) for k, v in dangling.items())
	dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

	# power iteration: make up to max_iter iterations
	for _ in range(max_iter):
		xlast = x
		x = dict.fromkeys(xlast.keys(), 0)
		danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
		for n in x:

			# this matrix multiply looks odd because it is
			# doing a left multiply x^T=xlast^T*W
			for nbr in W[n]:
				x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
			x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]


		# check convergence, l1 norm
		err = sum([abs(x[n] - xlast[n]) for n in x])
		if err < N*tol:
			return x
		raise NetworkXError('pagerank: power iteration failed to converge '
						'in %d iterations.' % max_iter)

	danglevector = [1 for n in W if W.out_degree(n, weight=weight) == 0.0 else 0]
	danglevector = np.array(danglevector)
	xlast, x, p = dict2nparray(xlast), dict2nparray(x), dict2nparray(p)
	for _ in range(max_iter):
		xlast = x
		x = np.zeros(N)
		x = alpha*(np.ones(N) * np.sum(xlast*danglevector)/N)+alpha*xlast + (1.0-alpha)*p
	

def dict2nparray(d):
	arr = [d[k] for k in sorted(list(d.keys()))]
	return np.array(arr)


def find_busiest_period(data):
  
  cnt = 0
  last_cnt = 0
  peak = 0
  busiest = None
  last = data[0][0]
  
  for cur, num, flag in data:  
    # 0: 1487799425: cnt: 14, 1: cnt: 10; 2: cnt: 8|3: 1487800378 : peak: 8, busiest: 1487799425, last-> 1487800378, cnt:18 
    # 
    if cur != last:
      if cnt > peak:
        peak = cnt
        busiest = last
      #last_cnt = cnt
      last = cur
    cnt += 2*(flag-0.5) *num
  
  return busiest


  function findBusiestPeriod(data):
    n = data.length
    count = 0
    maxCount = 0
    maxPeriodTime = 0

    for i from 0 to n-1:
        # update count
        if (data[i][2] == 1): # visitors entered the mall  
            count += data[i][1]
        else if (data[i][2] == 0): # visitors existed the mall
            count -= data[i][1]

        # check for other data points with the same timestamp
        if (i < n-1 AND data[i][0] == data[i+1][0]):
            continue

        # update maximum
        if (count > maxCount):
            maxCount = count
            maxPeriodTime = data[i][0]

    return maxPeriodTime

def find_busiest_period(data):
  
	data.append([0, -float("inf"), 0])
	cnt = 0
	last_cnt = 0
	peak = 0
	busiest = None
	last = data[0][0]
	
	for cur, num, flag in data:  
		# 0: 1487799425: cnt: 14, 1: cnt: 10; 2: cnt: 8|3: 1487800378 : peak: 8, busiest: 1487799425, last-> 1487800378, cnt:18 
		# 
		if cur != last:
		if cnt > peak:
			peak = cnt
			busiest = last
		#last_cnt = cnt
		last = cur
		cnt += 2*(flag-0.5) *num
	
	return busiest


print(find_busiest_period([[1487799425,21,1],[1487799425,4,0],[1487901318,7,0]]))


def check1():
    # Main
    # 1. Create a directed graph with N nodes
    g = nx.DiGraph()
    N = 5
    g.add_nodes_from(range(N))

    # 2. Add directed edges in graph
    g = add_edges(g, 0.4)

    # 3. perform a random walk
    points = random_Walk(g)

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

def check2():
    G=nx.barabasi_albert_graph(6,4)
    pr=pagerank(G,0.4)
    print(pr)

check2()


