"""
A simplied scenario can be described as a set of states and possible transitions between them.
They can be represented by an adj map:
adj_map = {0: [1, 3], 1: [4, 5, 3], 2: [4, 3], 3: [5], 4: [5], 5: []}
each state is an integer.
We would like to be able to tell whether given k steps, we can go from a start node to an end node within those steps

start = 0
end = 4
k = 5
--> True

start = 0
end = 4
k = 1
--> False

"""

#start = 0
#end = 5
#k = 1  
#adj_map = {0: [1, 5], 1: [2], 2: [3], 3:[4], 4: [5]#, 5:[]}
#


def find_path(adj_map, start, end, k): # O(|E|+|V|)
    
    visited = set([start])
    def helper(node, step):
        if step > k:
            return False
        
        if node == end and step <= k:
            return True
        
        for nei in adj_map[node]:
            if nei not in visited:
                visited.add(nei)
                res = helper(nei, step + 1)
                if res:
                    return True
        
        return False

    return helper(start, 0)

start = 0
end = 5
k = 2 
adj_map = {0: [1], 1: [2], 2: [3], 3:[4], 4: [5], 5:[]}
print(find_path(adj_map, start, end, k))




"""

adj_map=  { i, [(p(i->j), j), ...] } 

adj_map = {0: [(0.6, 1), (0.4, 3)], ...}

What is the probability of reaching end from start within k steps?

start = 0
end = 3
k = 1
--> 0.4
# P(s -> t) = \sum_{disjoint path} P(s -> path -> t) 0->1->2->3, 0->4->5->3

0->1->3     

innode(t)
P(s -> t) = \sum_ innode(t) P (s -> innode) * P(innode -> t)
P(s -> innode v) # iterative way

#adj_map = {0: [1, 2], 1: [2, 3], 2: [3], 3:[]}
s = 0
# each iteration, s-> all the possible nodes O(|V|) at the worst case
# number of innodes: O(|V|)
p(s->3, k) : p(s -> 2) p2 + p(s -> 1) p1
innode of 2: 1, 0
p(s -> 2) : p(s -> 1)*p + p(s -> 0)*p

p(s -> 1) : 

p(s->1, k).   

k * (|V| + |E|)^2

#adj_map = {0: [1, 2], 1: [2, 3], 2: [3], 3:[]}
#0 -> 3 in 10 steps
P(s->2, step)

P(s->2)
"""

"""
Let me rewrite the formula:
let's say v is an innode of u, then 
P(s -> u) = \sum_{innode of u} P(s -> v) * P(v -> u)

The concern of "cyclic updating": we add a parameter i (0<=i<=k) to denote the current steps we have taken.
then 
P( s-> t, i) = \sum_{innode of t} P(s -> t, i-1) * P(v -> u)
We start from computing all the P with i == 1 frist, then i == 2 and so on... until i == k. So there is no need to worry about any cyclic updates since we can't use P with more steps to update P with fewer steps.
A draft of the DP (have not been optimized)

for i in range(k):
    for s in V:
        for t in V:
            P[s][t][i] = sum(P[s][v][i-1]*A[v][t] for v in innodes(t))

return P[S][T][k]
"""