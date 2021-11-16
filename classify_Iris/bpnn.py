import tensorflow as tf
from pandas import read_csv
from sklearn import datasets, model_selection
import numpy as np
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

x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,random_state=1,test_size=0.3)
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
#模拟层，设置w,b参数
w = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1))
b = tf.Variable(tf.random.truncated_normal([3], stddev=0.1))
#设置学习率、轮数
lr = 0.1
epoch = 1000
loss_count= 0
acc=0
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w) + b
            y = tf.nn.softmax(y)
            #3层
            y_ = tf.one_hot(y_train, depth=3)
            #计算损失，即对应输出值与真实值误差
            loss = tf.reduce_mean(tf.square(y_ - y))
            loss_count += loss.numpy()
        #计算差值
        grads = tape.gradient(loss, [w, b])
        # 梯度更新
        w.assign_sub(lr * grads[0])
        b.assign_sub(lr * grads[1])
        #loss-a/4即为丢失率
    count, total= 0, 0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w) + b
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)
        pred = tf.cast(pred, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        count += int(correct)
        total += x_test.shape[0]
    acc = count / total
    loss_count=0
print("1000次学习后准确率", acc)

