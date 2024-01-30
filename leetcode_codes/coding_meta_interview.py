def say_hello():
    print('Hello, World')

for i in range(5):
    say_hello()


# Your previous Plain Text content is preserved below:

# Welcome to Meta!

# This is just a simple shared plaintext pad, with no execution capabilities.

# When you know what language you would like to use for your interview,
# simply choose it from the dropdown in the left bar.

# Enjoy your interview!

Hi, Yi!

# A = [1,2,3]
# B = [3,4,5] n2

# A = [1,100,300, _,_,_]


# assume we need to sort A[:n1]
# i,j = n1-1, n2-1

def merge_two_arr(arr1, arr2): #O(m+n)

    i, j =0, 0
    n1, n2 = len(arr1), len(arr2)
    res = []

    while i<n1 and j<n2: # 
        a1 = arr1[i]
        a2 = arr2[j]
        if a1<a2:
            res.append(a1)
            i += 1
        elif a2<a1:
            res.append(a2)
            j+=1
        else:
            res.append(a1)
            res.append(a2)
            i += 1
            j += 1
    
    # res= [1,2,3,3]
    if j<n2:
        for num in arr2[j+1:]:
            res.append(num)
    if i<n1:
        for num in arr1[i+1:]:
            res.append(num)
    return res

   

# k sorted array
#A = [2,3] 0
#B = [-2,3] 1
#C= [0,1,2] 2
# output = [-2,0,1,2,2,3,3]
#O(k)*max(all the length of k arrays).
# heap k length O(nlogk)

# initial: hq [[2,0,0], [2,2,2], [3, 1,1]]
#  [-2,1,0] push [3, 1,1]
# [0,2,0] push [1, 2, 1]
# [1, 2, 1] push [2,2,2]
# ....
#...

#res = [-2, 0, 1,2 ....]

#[-2,0,1,2,3]

def mergedK(arrs, k):

    hq = [[arrs[i][0], i, 0] for i in range(len(arrs))] #i: idx of arr in the list of arr, 0: idx 
    heapq.heapify(hq)
    res = []
    # O(K)
    while hq:
        cur_min, arr_idx, val_idx = heapq.heappop(hq) # -2, 1, 0
        res.append(cur_min) # if cur_min != res[-1]:  res.append(cur_min)
        nex_idx = val_idx + 1
        if nex_idx < len(arrs[arr_idx]):
            term2push = [arrs[arr_idx][nex_idx], arr_idx, nex_idx]
            heapq.heappush(hq, term2push)
    return res

    