def partition(arr, s: int, t: int)-> int:
	if s>t:
		return False
	i, j = s, s
	pivot = arr[t]
	for j in range(s, t):
		if arr[j] < pivot:
			arr[i] , arr[j] = arr[j], arr[i]
			i += 1
		arr[i+1], arr[t] = arr[t], arr[i+1]
	return arr, i+1

def quickselect(arr, k):

	L = len(arr)
	high = L-1
	low = 0

	if k>L:
		return None

	_, index = partition(arr, low, high)
	while index-1 != k-1:
		if index-1 == k-1:
			return arr[index]
		elif index-1 > k-1:
			high = index - 1
		elif index -1 < k-1:
			low = index + 1
			_, index = partition(arr,low,high)
	return arr[index]

def quickSort(arr, low, high):

    if low >= high:
        return []

    _, mid = partition(arr, low, high)
    
    if mid-1>=low:
        quickSort(low, mid-1)
    if mid+1<=high:
        quickSort(mid+1, high)


arr = [4,5,6,1,2,3]
res = quickselect(arr, 3)
res = quickSort(arr, 0, len(arr)-1)
print(res)