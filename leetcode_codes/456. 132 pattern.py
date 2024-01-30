from typing import List

def find132pattern(self, nums: List[int]) -> bool:
    s3 = float('-inf')  # equivalent to INT_MIN in C++
    st = []  # use a list as a stack in Python
    
    for i in range(len(nums) - 1, -1, -1):
        if nums[i] < s3:
            return True
        while st and nums[i] > st[-1]:  # st[-1] gives the top element of the stack
            s3 = st[-1]
            st.pop()
        st.append(nums[i])
    return False