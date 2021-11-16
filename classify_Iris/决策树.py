from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = read_csv('C:/Users/lenovo/Desktop/Iris.csv')
data1=np.asarray(list(df['PetalLengthCm']))
data2=np.asarray(list(df['PetalWidthCm']))
data3=np.asarray(list(df['SepalLengthCm']))
data4=np.asarray(list(df['SepalWidthCm']))
data5=np.asarray(list(df['Species']))
dat_l=[]
speices=[]
for i in range(len(data1)):
    data=[]
    data.append(data1[i])
    data.append(data2[i])
    data.append(data3[i])
    data.append(data4[i])
    if data5[i]=='Iris-setosa':
        speices.append(0)
    elif data5[i]=='Iris-versicolor':
        speices.append(1)
    else:
        speices.append(2)
    data=np.asarray(list(data))
    dat_l.append(data)
data=np.asarray(dat_l)
x=data
y=np.asarray(speices)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=3)
dt_model = DecisionTreeClassifier()
dt_model.fit(train_x, train_y)
predict= dt_model.predict(test_x)
score = dt_model.score(test_x, test_y)
print(predict)
print(test_y)
print('准确率：', score)