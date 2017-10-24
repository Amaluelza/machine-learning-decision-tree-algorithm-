from sklearn.datasets import load_iris
import numpy as np
from sklearn.tree import DecisionTreeClassifier
iris=load_iris()
print(iris.feature_names)
print(iris.target_names)
#print(iris.data[0])
#print(iris.target[0])
x=[0,50,100]
xtrain=np.delete(iris.data,x,axis=0) #deletes 0,50,100th row dataset axis=0 
#2D is converted to 1D so to remove errors
ytrain=np.delete(iris.target,x) #deletes target set from dataset
xtest=iris.data[x] #test set is given to xtest
ytest=iris.target[x]
clf=DecisionTreeClassifier()
clf.fit(xtrain,ytrain)
print(ytest)
print("prediction=",clf.predict(xtest))
