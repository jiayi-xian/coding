import numpy as np

def missing_val():
    pass

def find_outliner():
    pass

def train(X, y, label, iter):
    # linear regression, logistic regression, SVM, DT, 
    pass




def accuracy(y, label):

    y = np.array(y)
    label = np.array(label)

    accu = np.sum(y == label)/len(label)
    return acc
    
    
    
def minMeetingRooms(self, intervals):
    res = cur = 0
    for i, v in sorted(x for i,j in intervals for x in [[i, 1], [j, -1]]):
        cur += v
        res = max(res, cur)
    return res

"""N = 5, tickets = [(1, 3), (2, 4)]
dict = {1: 1, 2: 1, 3: -1, 4: -1}
timestamp = timestamp[i - 1] + dict if i in dict and dict > 0 else timestamp[i - 1]"""

def f(N, tickets):
    timestamps = list(range(N))
    cars = 0
    # swiping lines
    times = sorted(x for i,j in tickets for x in [[i, 1], [j, -1]]) # O(nlogn)

