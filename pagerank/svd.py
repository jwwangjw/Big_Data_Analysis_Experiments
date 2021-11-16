import numpy as np
from fractions import Fraction


f=Fraction(1,3)
tran_mat=np.mat([[0,0.5,0,0.5],[f,0,0,0.5],[f,0.5,0,0],[f,0,1,0]])
pr=np.mat([[0.25],[0.25],[0.25],[0.25]])
print("未迭代：")
print("A:")
print(pr[0,0])
print('B:')
print(pr[1,0])
print("C:")
print(pr[2,0])
print("D:")
print(pr[3,0])
pr=tran_mat*pr
for i in range(10):
    print("第"+str(i+1)+"次迭代：")
    print("A:")
    print(pr[0,0])
    print('B:')
    print(pr[1,0])
    print("C:")
    print(pr[2,0])
    print("D:")
    print(pr[3,0])
    pr=tran_mat*pr
