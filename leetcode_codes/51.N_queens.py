from typing import List
import math
def solveNQueens(n: int):
    
    boards = []
    limit = (1 << n) - 1  # 这是怎么想的？
            
    def backtrack(row, path, ldiag, rdiag, board):
        if (path & limit) == limit:
            boards.append(board)
            return 1

        # 对当前行i 尝试摆放皇后于列j
        ban = path | ldiag | rdiag
        candidate = (~ban) & limit # (~ban) & n doesn't work

        ans = 0 # number of ways to place queen
        while candidate != 0: # exhaut all the potential options
            place = (-candidate) & candidate # get the least significate 1
            candidate ^= place
            ans += backtrack(row + 1, path | place, (ldiag | place) << 1, (rdiag| place) >> 1, board + [place])
        
        return ans
    number = backtrack(0, 0, 0, 0, [])
    queen_boards = []
    for board in boards:
        queens = []
        for loc in board:
            row = [""]*n
            row[int(math.log2(loc))] = "Q"
            queens.append(row)
        
        queen_boards.append(queens)

    return queen_boards

res = solveNQueens(4)
print("OK")
