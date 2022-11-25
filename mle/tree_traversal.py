# A binary tree node
class Node:

	# Constructor to create a new node
	def __init__(self, data):
		self.data = data
		self.left = None
		self.right = None

def iterativeInorder(root):

    q = []
    res = []
    cur = root
    
    while q or cur:
        while cur:
            q.append(cur)
            cur = cur.left
        
        cur = q.pop()
        res.append(cur.data)
        cur = cur.right
    return res



# An iterative process to print preorder traversal of BT
def iterativePreorder(root):
    q = []
    res = []
    cur = root
    while q or cur:
        if cur is None:
            cur = q.pop()
        res.append(cur.data)
        if cur.right:
            q.append(cur.right)
        cur = cur.left
    return res
	
# Driver program to test above function
root = Node(10)
root.left = Node(8)
root.right = Node(2)
root.left.left = Node(3)
root.left.right = Node(5)
root.right.left = Node(2)
res1 = iterativePreorder(root)
res2 = iterativeInorder(root)
print(res1)
print(res2)
from ppbtree import *
print_tree(root, nameattr='data', left_child='right', right_child='left')