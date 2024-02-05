class Treecur:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def inorderTraversal_interative(root):
    res = []
    stack = []
    cur = root
    stack_ = []
    while stack or cur:
        # cur = stack[-1] # 重要 如何使得不重复scan left 从stack里面出来的cur 只考虑right
        """
        while cur and cur.left:
            stack.append(cur.left) # 如果下面接continue 会继续在内循环里面打转
            stack_.append(cur.left.val)
            cur = cur.left
        """
        while cur:
            stack.append(cur) # 如果下面接continue 会继续在内循环里面打转
            stack_.append(cur.val)
            cur = cur.left

        cur = stack.pop()
        val = stack_.pop()
        res.append(cur.val)

        cur = cur.right # 重要 如何使得不重复scan left 从stack里面出来的cur 只考虑right

    return res

def postorderTraversal_iterative(root): # left -> right -> root
    res = []
    stack = []
    stack_ = []
    cur = root
    pre = None
    while stack or cur:
        while cur:
            stack.append(cur)
            stack_.append(cur.val)
            cur = cur.left

        # 1. 用 cur is None 来控制要不要考虑left child (循环1)
        # 2. 用 pre != cur.right 和 right is None 来控制要不要考虑right child (循环2)
        # cur is None
        # if cur is from while loop, thne stack[-1] does not have left child

        cur = stack[-1]
        if cur.right != pre and cur.right:
            cur = cur.right
            continue
        elif cur.right == pre:
            res.append(cur.val)
            stack.pop()
            stack_.pop()
            pre = cur
            cur = None          
        else: # cur 是 None 下一轮继续取 stack[-1].right
            res.append(cur.val)
            stack.pop()
            stack_.pop()
            pre = cur
            cur = cur.right #直接使用cur = None 和中间的case 统一就好

    return res


# Construct the tree as per the given image

def tree_construct():
    root = Treecur(1)
    root.left = Treecur(7)
    root.right = Treecur(9)
    root.left.left = Treecur(2)
    root.left.right = Treecur(6)
    root.left.right.left = Treecur(5)
    root.left.right.right = Treecur(11)
    root.right.right = Treecur(9)
    root.right.right.left = Treecur(5)
    
    return root

root = tree_construct()
res = inorderTraversal_interative(root)
res = postorderTraversal_iterative(root)