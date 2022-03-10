import numpy as np
def quant(lines):
    ans = [np.zeros((8,8)), np.zeros((8,8))]
    line = 0
    for j in range(8):
        for i in range(2):
            for k in range(8):
                ans[i][j][k] = int(lines[line])
                line +=1
    return ans

def print_array(array):
    print('np.array([',end='')
    for r in range(len(array)):
        print('[',end='')
        for i in range(len(array[r])):
            print(int(array[r][i]), end='')
            if i < len(array[r])-1:
                print(',', end='')
        print(']',end='')
        if r < len(array)-1:
            print(',', end='')
    print('])')
    
lns = open('aid').readlines()
lns = [l[:-1] for l in lns]
ans = quant(lns)
print_array(ans[0])
print_array(ans[1])
