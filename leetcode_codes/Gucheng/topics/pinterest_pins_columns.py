"""
Pinterest app screen is two columns of images (pins).
Each pin has a top and bottom index (top < bottom), and has a column represented by "L" or "R".
Given an unsorted list of non-overlapping pins like

pins = [(top_ind,bottom_ind,column),...,]
and a screen_length of screen_len
Return the maximum number of pins the user can possibly see (fully).
That is, complete this:


def get_max_pins(pins,screen_len):
    max_pins = 0
	...
    return max_pins
Example:

input:
  pins = [(1,4,"L"),(2,3,"R"),(4,8,"R"),(6,9,"L")] 
  screen_len = 5
output: 2
"""

def solution(pins, L):

    lpins = [pin for pin in pins if pin[2]== "L"]
    rpins = [pin for pin in pins if pin[2]== "R"]
    cur_t = pins[0][0]
    cur_b = cur_t + L
    j0, j1, k0, k1= 0, 0, 0, 0
    max_cnt = 0
    lpt, rpt = 0, 0

    for i in range(len(pins)):
        cur_t = pins[i][0]
        cur_b = cur_t + L
        flag = pins[i][2]
        lpt += flag == "L"
        rpt += flag == "R"
        cur_cnt = 0
        

        if flag == "R":
            while j0<=len(lpins)-1 and lpins[j0][0] < cur_t: # and pins[j0][2] == "L" count L pins
                j0 += 1
            j1 = max(j1, j0)
        else:
            j0 = lpt -1
            j1 = j0
        while j1 <= len(lpins)-1 and lpins[j1][1] <= cur_b:
            j1 += 1
        
        cur_cnt += max(0, j1 - j0)

        if flag == "L": 
            while k0<=len(rpins)-1 and rpins[k0][0] < cur_t:
                k0 += 1
            k1 = max(k1, k0)
        else:
            k0 = rpt -1
            k1 = k0
        while k1 <= len(rpins)-1 and rpins[k1][1] <= cur_b:
            k1 += 1
        
        cur_cnt += max(0, k1 - k0)

        max_cnt = max(max_cnt, cur_cnt)
        print(cur_t, cur_b, j0, j1, k0, k1, cur_cnt)
    
    print(max_cnt)
    print(lpins, "\n", rpins)
    return max_cnt


pins = [(1,4,"L"),(2,3,"R"),(4,8,"R"),(6,9,"L")] 
screen_len = 5
pins = [(1,4,"L"),(2,3,"R"),(4,10,"R"),(5,6,"L")]
screen_len = 7
pins = [(1,4,"L"),(2,3,"R"),(4,15,"R"),(5,6,"L")]
screen_len = 7
pins = [(1,4,"L"),(1,1,"R"),(2,2,"R"),(3,3,"R"), (4,5, "L")]
screen_len = 2
solution(pins, screen_len)