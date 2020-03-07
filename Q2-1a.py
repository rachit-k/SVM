import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from cvxopt import solvers
from cvxopt import matrix
import timeit


with open("train.csv") as f1:
    x = np.genfromtxt(f1, delimiter=',')

#print(x.shape) #(22500, 785)
#print(x[:,-1])
y=x[:,-1]
x=x[:,:-1]
d1=x[(y==4,)]
d2=x[(y==5,)]
#y1=y[(y==4)]
#y2=y[(y==5)]
#y=np.concatenate((y1, y2), axis=0)

#print(d2.shape) #(2250, 785)
#print(d2.shape) #(2250, 784)
#d1=d1/255.0
d1=d1*(-1)/255.0
d2=d2/255.0

X=np.concatenate((d1, d2), axis=0)
#print(X.shape) #(4500, 784)
n1=d1.shape[0]
n2=d2.shape[0]
m=n1+n2
Y=np.concatenate((np.ones((n1,1))*(-1.0),np.ones((n2,1))),axis=0)
P=np.dot(X,X.T)#*np.outer(Y,Y)
q=np.ones((m,1))*(-1)
G=np.concatenate((np.identity(m),(np.identity(m))*(-1.0)),axis=0)
h=np.concatenate((np.ones((m,1)),np.zeros((m,1))),axis=0)
A=Y.T #y is in this type

b=np.array([[0.0]])

P=matrix(P, tc='d')
q=matrix(q, tc='d')
G=matrix(G, tc='d')
h=matrix(h, tc='d')
A=matrix(A, tc='d')
b=matrix(b, tc='d')

start_time = timeit.default_timer()
sol=solvers.qp(P,q,G,h,A,b)
end_time = timeit.default_timer()

sol=sol['x']
#print(sol)

w=np.dot(sol.T,X)

#b1=0
#b2=0
#for i in range(m):
#    if(i<n1):
#        if(np.dot(w,d1[i].T)<b1):
#            b1=np.dot(w,d1[i].T)
#    else:
#        if(np.dot(w,d1[i-n1].T)>b2):
#            b2=np.dot(w,d2[i-n1].T)              
#print((b1+b2)/2)


for i in range(m):
    if(sol[i]>0.00001):
        if(i<n1):
            b=-1+np.dot(w,d1[i].T)
            break
        else:
            b=1-np.dot(w,d2[i-n1].T)
            break
print(b)
with open("test.csv") as f1:
    x = np.genfromtxt(f1, delimiter=',')
y=x[:,-1]
x=x[:,:-1]
td1=x[(y==4,)]
td2=x[(y==5,)]
#y1=y[(y==4)]
#y2=y[(y==5)]
#y=np.concatenate((y1, y2), axis=0)

td1=td1/255.0
td2=td2/255.0

t1=td1.shape[0]
t2=td2.shape[0]
t=t1+t2

correct=0.0

for i in range(t):
    if(i<t1):
        if (((np.dot(w,td1[i].T))+b)<0.0):
            correct=correct+1 
        else:
            print(i)
    else:
        if (((np.dot(w,td2[i-t1].T))+b)>0.0):
            correct=correct+1
        else:
            print(i)

print("correct:")
print(correct)
print("total:")
print(t)
print('accuracy:')  
print(correct*100/t)


