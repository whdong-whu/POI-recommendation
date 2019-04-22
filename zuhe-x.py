# encoding=utf-8

import numpy  as np

x1=np.loadtxt('x1.txt')
x2=np.loadtxt('x2.txt')
x3=np.loadtxt('x3.txt')
x4=np.loadtxt('x4.txt')
x5=np.loadtxt('x5.txt')
x6=np.loadtxt('x6.txt')
x7=np.loadtxt('x7.txt')
x8=np.loadtxt('x8.txt')
x=np.vstack((x1,x2,x3,x4,x5,x6,x7,x8))
np.savetxt("x.txt",x)
