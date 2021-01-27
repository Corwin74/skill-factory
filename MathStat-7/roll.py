WINDOW_LENGHT = 3
x = [1, 0, 1, -1, 2, 0, 1]
for ii in range(0,len(x)-WINDOW_LENGHT+1):
    print(x[ii]*0.5+x[ii+1]*0.3+x[ii+2]*0.2)
