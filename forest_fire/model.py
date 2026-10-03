import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera
import random

m = 50
n = 50
t = 100
k = 1
f = 0.01
p = 0.01

def loc(i,j,mat):
    s00 = mat[i-1,j-1]
    s01 = mat[i-1,j]
    s02 = mat[i-1,(j+1)%n]
    s10 = mat[i,j-1]
    #s11 = mat[i,j]
    s12 = mat[i,(j+1)%n]
    s20 = mat[(i+1)%m,j-1]
    s21 = mat[(i+1)%m,j]
    s22 = mat[(i+1)%m,(j+1)%n]

    if i-1 < 0:
        s00 = 0
        s01 = 0
        s02 = 0

    if i+1 > m-1:
        s20 = 0
        s21 = 0
        s22 = 0

    if j-1 < 0:
        s00 = 0
        s10 = 0
        s20 = 0

    if j+1 > n-1:
        s02 = 0
        s12 = 0
        s22 = 0

    d = [s00,s01,s02,s10,s12,s20,s21,s22]

    o = 0
    for g in d:
        if g == 2:
            o += 1

    return o    

mat = np.random.randint(0, 2, size=(m, n))
mat_c = mat.copy()
mat[10,10] = 2
mat[10,11] = 2

fig = plt.figure()
camera = Camera(fig)
for o in range(t):
    if o > 0:
        for i in range(m):
            for j in range(n):
                r1 = random.random()
                r2 = random.random()
                if mat[i,j] == 2:
                    mat_c[i,j] = 0
                elif mat[i,j] == 0:
                    if (loc(i,j,mat) == 0)and(r1 < p):
                        mat_c[i,j] = 1
                    else:
                        mat_c[i,j] = 0
                elif mat[i,j] == 1:
                    if r2 < f:
                        mat_c[i,j] = 2
                    else:
                        if loc(i,j,mat) >= k:
                            mat_c[i,j] = 2
                        else:
                            mat_c[i,j] = 1
              
        mat = mat_c.copy()
    plt.imshow(mat,cmap='viridis')
    camera.snap()
animation = camera.animate()

plt.show()
