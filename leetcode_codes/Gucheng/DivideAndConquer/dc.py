from typing import *
def reversePairs(nums: List[int]) -> int:

    arr = nums
    def dc(arr, s, t):
        if s == t:
            return 0
        
        m = s + (t-s)//2
        return dc(arr, s, m) + dc(arr, m+1, t) + merge(arr, s, m, t)

    
    def merge(arr, l, m, r):
        
        help_arr = []
        ans = 0
        rPj = 0
        j, a, b = m+1, l, m+1

        for i in range(l, m+1):
            while j <= r and arr[i] > 2*arr[j]: 
                rPj += 1
                j += 1
            ans += rPj

        while a<=m and b<=r:

            if arr[a]<=arr[b]:
                help_arr.append(arr[a])
                a += 1
            elif arr[a] > arr[b]:
                help_arr.append(arr[b])
                b += 1
            
        if a<=m:
            help_arr.extend(arr[a:m+1])
        elif b<=r:
            help_arr.extend(arr[b:r+1])
        arr[l:r+1] = help_arr

        return ans

    return dc(arr, 0, len(arr)-1)

print(reversePairs([1,3,2,3,1]))

            
        
