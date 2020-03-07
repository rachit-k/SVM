import numpy as np
from sklearn.svm import SVC
import timeit
import matplotlib.pyplot as plt

with open("train.csv") as f1:
    x = np.genfromtxt(f1, delimiter=',')

#print(x.shape) #(22500, 785)
#print(x[:,-1])
y=x[:,-1]
X=x[:,:-1]
X=X/255.0
c_list=[ 0.00001 , 0.001, 1, 5, 10]

m=X.shape[0]
starttime = timeit.default_timer()
vscores=[]
tscores=[]

with open("test.csv") as f1:
    testx = np.genfromtxt(f1, delimiter=',')
testy=testx[:,-1]
testX=testx[:,:-1]
testX=testX/255.0

for c in c_list:
    clf = SVC(kernel='rbf', gamma=0.05, C=c ,decision_function_shape='ovo')
    score=0
    for i in range(5):
        X1=np.delete(X,list(range((i%5), X.shape[0], 5)), axis=0)
        y1=np.delete(y,list(range((i%5), y.shape[0], 5)), axis=0)
        tX1=X[(i%5)::5,:]
        tY1=y[(i%5)::5]
        clf.fit(X1,y1)
        score=score+clf.score(tX1,tY1)
    score=score/5
    vscores.append(score)
    clf.fit(X,y)
    tscores.append(clf.score(testX,testy))
    
print("time:")    
print(timeit.default_timer()-starttime)

print(vscores)
print(tscores)   
plt.plot(c_list, vscores,'-o',label='validation scores')  
plt.plot(c_list, tscores,'-o',label='test scores')
plt.savefig("Q2-2d.png",bbox_inches="tight")
plt.show()

#time:
#13104.124432333001
#[0.09288888888888888, 0.09288888888888888, 0.8792888888888889, 0.8839555555555556, 0.8836444444444445]
#[0.5736, 0.5736, 0.8808, 0.8828, 0.8824]