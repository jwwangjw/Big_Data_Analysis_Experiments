import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn import model_selection
from sklearn.svm import SVC

df = read_csv('C:/Users/lenovo/Desktop/Iris.csv')
data1=np.asarray(list(df['PetalLengthCm']))
data2=np.asarray(list(df['PetalWidthCm']))
data3=np.asarray(list(df['SepalLengthCm']))
data4=np.asarray(list(df['SepalWidthCm']))
data5=np.asarray(list(df['Species']))
dat_l=[]
for i in range(len(data1)):
    data=[]
    data.append(data1[i])
    data.append(data2[i])
    data.append(data3[i])
    data.append(data4[i])
    if data5[i]=='Iris-setosa':
        data.append(0)
    elif data5[i]=='Iris-versicolor':
        data.append(1)
    else:
        data.append(2)
    data=np.asarray(list(data))
    dat_l.append(data)
data=np.asarray(dat_l)

X ,y=np.split(data,(4,),axis=1)
x=X[:,0:2]
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,random_state=1,test_size=0.3)

clf=SVC(kernel='rbf',gamma=0.1,decision_function_shape='ovo',C=0.8)
clf.fit(x_train,y_train.ravel())
print(clf.score(x_test,y_test))
