import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix 
import timeit

with open("train.csv") as f1:
    x = np.genfromtxt(f1, delimiter=',')

#print(x.shape) #(22500, 785)
#print(x[:,-1])
y=x[:,-1]
X=x[:,:-1]
d1=X[(y==4,)]
d2=X[(y==5,)]
y1=y[(y==4)]
y2=y[(y==5)]
y=np.concatenate((y1, y2), axis=0)
X=np.concatenate((d1, d2), axis=0)
X=X/255.0
m=X.shape[0]
starttime = timeit.default_timer()
clf = SVC(kernel='linear')

#clf = SVC(kernel='linear', decision_function_shape='ovo')#0.836
#clf = SVC(kernel='rbf', gamma=0.05)
#clf = SVC(kernel='rbf', gamma=0.05 ,decision_function_shape='ovo')
clf.fit(X,y)
print(timeit.default_timer()-starttime)

with open("test.csv") as f1:
    x = np.genfromtxt(f1, delimiter=',')
y=x[:,-1]
X=x[:,:-1]
d1=X[(y==4,)]
d2=X[(y==5,)]
y1=y[(y==4)]
y2=y[(y==5)]
y=np.concatenate((y1, y2), axis=0)
X=np.concatenate((d1, d2), axis=0)
X=X/255.0

print(clf.score(X,y))
print(confusion_matrix(y,clf.predict(X)))
#print(clf.intercept_)