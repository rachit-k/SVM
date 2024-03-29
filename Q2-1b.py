import numpy as np
import math
import timeit
from cvxopt import solvers
from cvxopt import matrix
import scipy.spatial.distance as spdist


with open("train.csv") as f1:
    x = np.genfromtxt(f1, delimiter=',')

#print(x.shape) #(22500, 785)
y=x[:,-1]
x=x[:,:-1]/255.0
x1=x[(y==4,)]
x2=x[(y==5,)]

#print(d2.shape) #(2250, 784)

X=np.concatenate((x1, x2), axis=0)
#print(X.shape) #(4500, 784)
n1=x1.shape[0]
n2=x2.shape[0]
m=n1+n2
Y=np.concatenate((np.ones((n1,1))*(-1.0),np.ones((n2,1))),axis=0)

starttime = timeit.default_timer()

dist= spdist.squareform(spdist.pdist(X,'euclidean'))

gamma=0.05
dist=np.exp(-1*gamma*dist*dist)
P=dist*np.dot(Y,Y.T)

q=np.ones((m,1))*(-1.0)
G=np.concatenate((np.diag(np.ones(m)),np.diag(np.ones(m)*(-1.0))),axis=0)
h=np.concatenate((np.ones((m,1)),np.zeros((m,1))),axis=0)
A=Y.T
b=np.array([[0]])

P=matrix(P, tc='d')
q=matrix(q, tc='d')
G=matrix(G, tc='d')
h=matrix(h, tc='d')
A=matrix(A, tc='d')
b=matrix(b, tc='d')

sol=solvers.qp(P,q,G,h,A,b)


sol=sol['x']
#print(sol)
b=0
n_support=np.zeros((2,))
for i in range(m):
    if(sol[i]>0.00001):
        if(i<n1):
            b=-1
            for j in range(m):
                if(sol[j]>0.00001):
                    b=b-sol[j]*Y[j][0]*(math.exp(-1*gamma*((spdist.euclidean(X[j],X[i]))**2)))
            break
        else:
            b=1
            for j in range(m):
                if(sol[j]>0.00001):
                    b=b-sol[j]*Y[j][0]*(math.exp(-1*gamma*((spdist.euclidean(X[j],X[i]))**2)))
            break
        
for i in range(m):
    if(sol[i]>0.00001):
        if(i<n1):
            n_support[0]=n_support[0]+1
        else:
            n_support[1]=n_support[1]+1        
print("training time:")
print(timeit.default_timer()-starttime)
print('b:')
print(b)
print('no of support vectors:')
print(n_support)


with open("test.csv") as f1:
    x = np.genfromtxt(f1, delimiter=',')
y=x[:,-1]
x=x[:,:-1]/255.0
x1=x[(y==4,)]
x2=x[(y==5,)]

t1=x1.shape[0]
t2=x2.shape[0]
t=t1+t2
X1=np.concatenate((x1, x2), axis=0)
correct=0.0
l=[]

for i in range(t):
    if(i<t1):
        k=0
        for j in range(m):
                if(sol[j]>0.00001):
                    k=k+sol[j]*Y[j][0]*(math.exp(-1*gamma*((spdist.euclidean(X[j],X1[i]))**2)))
        if (k+b<0.0):
            correct=correct+1 
        else:
            print(i)
    else:
        k=0
        for j in range(m):
                if(sol[j]>0.00001):
                    k=k+sol[j]*Y[j][0]*(math.exp(-1*gamma*((spdist.euclidean(X[j],X1[i]))**2)))
        if (k+b>0.0):
            correct=correct+1
        else:
            print(i)
print('correct:')            
print(correct)
print("total:")
print(t)
print('accuracy:')
print(correct*100/t)
