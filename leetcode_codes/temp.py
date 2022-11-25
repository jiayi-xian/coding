import numpy as np

board = [[1,3,5,5,2],[3,4,3,3,1],[3,2,4,5,2],[2,4,4,5,5],[1,4,4,1,1]]  
a = zip(*reversed(board))  
transpose = list(map(list, zip(*board))) 
transpose_back = list(map(list, zip(*board)))  
transpose = list(map(list,zip(*reversed(board))))  




# rotate clockwise 90 degree back = list(reversed(list(zip(*board)))) # rotate counter-clockwise 90 degree    



