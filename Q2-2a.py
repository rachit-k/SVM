import numpy as np
import math
from cvxopt import solvers
from cvxopt import matrix
import scipy.spatial.distance as spdist
from sklearn.metrics import confusion_matrix 
import timeit

with open("train.csv") as f1:
    x = np.genfromtxt(f1, delimiter=',')

y=x[:,-1]
x=x[:,:-1]
x=x/255.0
m=x.shape[0]
gamma=0.05


def func(mm,nn):
    x1=x[(y==mm,)]
    x2=x[(y==nn,)]
    n1=x1.shape[0]
    n2=x2.shape[0]
    m=n1+n2
    
    X=np.concatenate((x1, x2), axis=0)    
    Y=np.concatenate((np.ones((n1,1))*(-1.0),np.ones((n2,1))),axis=0)
    q=np.ones((m,1))*(-1)
    G=np.concatenate((np.diag(np.ones(m)),np.diag(np.ones(m)*(-1.0))),axis=0)
    h=np.concatenate((np.ones((m,1)),np.zeros((m,1))),axis=0)
    B=np.array([[0]])    

    dist= spdist.squareform(spdist.pdist(X,'euclidean'))
    dist=np.exp(-1*gamma*dist*dist)
    P=dist*np.dot(Y,Y.T)
    A=Y.T
    q=matrix(q, tc='d')
    G=matrix(G, tc='d')
    h=matrix(h, tc='d')
    B=matrix(B, tc='d')
    A=matrix(A, tc='d')
    P=matrix(P, tc='d')

    sol=solvers.qp(P,q,G,h,A,B)

    print(sol)
    sol=sol['x']

    b=0
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
    return sol,b   


sols=[]
bs=[]
#k=0
starttime = timeit.default_timer()
for i in range(10):
    for j in range(i+1,10):
        a,b=func(i,j)
        sols.append(a)
        bs.append(b)
#        k=k+1
endtime = timeit.default_timer()
with open("test.csv") as f1:
    tx = np.genfromtxt(f1, delimiter=',')

ty=tx[:,-1]
tx=tx[:,:-1]
tx=tx/255.0
t=tx.shape[0]
       
correct=0.0
y_pred=np.zeros((t,))
for i in range(t):
    kk=0
    pred=np.zeros((10,))
    score=np.zeros((10,))
    for l in range(10):
        for n in range(l+1,10):
            d1=x[(y==l,)]
            d2=x[(y==n,)]
            n1=d1.shape[0]
            n2=d2.shape[0]
            m=n1+n2
            X=np.concatenate((d1, d2), axis=0)  
            Y=np.concatenate((np.ones((n1,1))*(-1.0),np.ones((n2,1))),axis=0)
            sol=sols[kk]
            b=bs[kk] 
            k=0
            for j in range(m):
                    if(sol[j]>0.00001):
                        k=k+sol[j]*Y[j][0]*(math.exp(-1*gamma*((spdist.euclidean(X[j],tx[i]))**2)))
            if (k+b<0.0):
                pred[l]=pred[l]+1
                score[l]=score[l]+(1/(1+math.exp(-1*abs(k+b))))
                score[n]=score[n]+(1-(1/(1+math.exp(-1*abs(k+b)))))
            else:
                pred[n]=pred[n]+1   
                score[n]=score[n]+(1/(1+math.exp(-1*abs(k+b))))
                score[l]=score[l]+(1-(1/(1+math.exp(-1*abs(k+b)))))
            
            kk=kk+1
    max= np.amax(pred)
    qq=[]
    print(i)
    for q in range(10):
        if(max==pred[q]):
            qq.append(q)
              
    q=qq[0]
    max=score[q]    
    for l in qq:
        if(max<score[l]):
            q=l
            max=score[q]
        
    y_pred[i]=q
    if(q==ty[i]):
        correct=correct+1
        print(correct)
print("training time:")
print(endtime-starttime)        
print("correct:") 
print(correct) 
print(correct*100/t) 
print("confusion matrix:")
print(confusion_matrix(ty,y_pred))
#training time:
#1967.2091065030002
#correct:
#4274.0
#85.48
#confusion matrix:
#[[463   0   7   8   1   0  11   0  10   0]
# [  2 484   4   9   0   0   1   0   0   0]
# [ 13   0 423   7  24   0  22   0  11   0]
# [ 25   2   3 457   3   0   5   0   5   0]
# [  6   1  74  63 308   0  40   0   8   0]
# [  0   0   0   0   0 474   0  16   5   5]
# [162   0  69  10  18   0 231   0  10   0]
# [  0   0   0   0   0  14   0 471   1  14]
# [  1   0   1   2   0   2   3   2 489   0]
# [  0   0   0   0   0  11   0  14   1 474]]     
