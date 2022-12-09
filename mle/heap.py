"""
# 堆中父节点i与子节点left、right的位置关系
parent = int((i-1) // 2)    # 取整
left = 2 * i + 1
right = 2 * i + 2

https://blog.csdn.net/qq_23869697/article/details/82735088

"""

class Array(object):
    """
    Achieve an Array by Python list
    """
    def __init__(self, size = 32):
        self._size = size
        self._items = [None] * size

    def __getitem__(self, index):
        """
        Get items
        :param index: get a value by index
        :return: value
        """
        return self._items[index]

    def __setitem__(self, index, value):
        """
        set item
        :param index: giving a index you want to teset
        :param value: the value you want to set
        :return:
        """
        self._items[index] = value

    def __len__(self):
        """
        :return: the length of array
        """
        return self._size

    def clear(self, value=None):
        """
        clear the Array
        :param value: set all value to None
        :return: None
        """
        for i in range(self._size):
            self._items[i] = value

    def __iter__(self):
        for item in self._items:
            yield item
            
class MaxHeap(object):
    def __init__(self, maxsize=None):
        self.maxsize = maxsize
        self._elements = Array(maxsize)
        self._count = 0

    def __len__(self):
        return self._count

    def add(self, value):
        if self._count >= self.maxsize:
            raise Exception('full')
        self._elements[self._count] = value
        self._count += 1
        self._siftup(self._count-1)  # 维持堆的特性

    def _siftup(self, ndx):
        if ndx > 0:
            parent = int((ndx-1)/2)
            if self._elements[ndx] > self._elements[parent]:    # 如果插入的值大于 parent，一直交换
                self._elements[ndx], self._elements[parent] = self._elements[parent], self._elements[ndx]
                self._siftup(parent)    # 递归

    def extract(self):
        if self._count <= 0:
            raise Exception('empty')
        value = self._elements[0]    # 保存 root 值
        self._count -= 1
        self._elements[0] = self._elements[self._count]    # 最右下的节点放到root后siftDown
        self._siftdown(0)    # 维持堆特性
        return value

    def _siftdown(self, ndx):
        left = 2 * ndx + 1
        right = 2 * ndx + 2
        # determine which node contains the larger value
        largest = ndx
        if (left < self._count and     # 有左孩子
                self._elements[left] >= self._elements[largest] and
                self._elements[left] >= self._elements[right]):  # 原书这个地方没写实际上找的未必是largest
            largest = left
        elif right < self._count and self._elements[right] >= self._elements[largest]:
            largest = right
        if largest != ndx:
            self._elements[ndx], self._elements[largest] = self._elements[largest], self._elements[ndx]
            self._siftdown(largest)


def test_maxheap():
    import random
    n = 5
    h = MaxHeap(n)
    for i in range(n):
        h.add(i)
    for i in reversed(range(n)):
        assert i == h.extract()

# 构造大顶堆，从非叶子节点开始倒序遍历，因此是l//2 -1 就是最后一个非叶子节点
l = len(arr)
for i in range(l//2-1, -1, -1): 
     build_heap()
     # 遍历针对每个非叶子节点构造大顶堆

class Solution(object):
    def heap_sort(self, nums):
        i, l = 0, len(nums)
        self.nums = nums
        # 构造大顶堆，从非叶子节点开始倒序遍历，因此是l//2 -1 就是最后一个非叶子节点
        for i in range(l//2-1, -1, -1): 
            self.build_heap(i, l-1)
        # 上面的循环完成了大顶堆的构造，那么就开始把根节点跟末尾节点交换，然后重新调整大顶堆  
        for j in range(l-1, -1, -1):
            nums[0], nums[j] = nums[j], nums[0]
            self.build_heap(0, j-1)

        return nums

    def build_heap(self, i, l): 
        """构建大顶堆"""
        nums = self.nums
        left, right = 2*i+1, 2*i+2 ## 左右子节点的下标
        large_index = i 
        if left <= l and nums[i] < nums[left]:
            large_index = left

        if right <= l and nums[left] < nums[right]:
            large_index = right
 
        # 通过上面跟左右节点比较后，得出三个元素之间较大的下标，如果较大下表不是父节点的下标，说明交换后需要重新调整大顶堆
        if large_index != i:
            nums[i], nums[large_index] = nums[large_index], nums[i]
            self.build_heap(large_index, l)

def heapify(arr, n, i):
    largest = i
    l = 2 * i + 1 # left = 2*i + 1
    r = 2 * i + 2 # right = 2*i + 2
    if l < n and arr[i] < arr[l]:
        largest = l
    if r < n and arr[largest] < arr[r]:
        largest = r
    if largest != i:
        arr[i] ,arr[largest] = arr[largest] ,arr[i]
        # 父节点与子节点交换位置之后，很可能该子节点所属的子树，又不符合大顶堆的规则了
        # 所以要对该子节点再做一次heapify，使其符合大顶堆的规则
    heapify(arr, n, largest)
    print(arr)

def heapSort(arr):
    n = len(arr)
    # Build a maxheap.
    for i in range(n//2-1, -1, -1): 
        # 在arr中非叶子节点的索引是从 n//2-1 到 0.在构建最大堆时，只需要遍历这些非叶子节点即可
        heapify(arr, n, i)
        # 一个个交换元素
    for i in range( n -1, 0, -1):
        # 在第一次循环时，现将上面构建好的大顶堆arr中的第一个和最后一个元素互换，
        # 相当于将arr中的最大元素放在了整个序列的最后。
        #　之后依次循环，每个循环都能找出当前序列的最大元素，并将其放到序列的最后。
        # 当剩下最后一个元素时，就是arr中最小的元素，这时候的arr就是排好序的序列了
        arr[i], arr[0] = arr[0], arr[i] # 交换
        heapify(arr, i, 0)

arr = [ 12, 11, 13, 5, 6, 7]
heapSort(arr)
n = len(arr)
print ("排序后")
for i in range(n):
    print ("%d" %arr[i]),