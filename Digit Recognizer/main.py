import pandas as pd
import tensorflow as tf
import numpy as np

n_classes=10
batch_size=100
epochs=300
learning_rate=0.0001

data=pd.read_csv('./data/train.csv')
train_data=data.iloc[:,1:]
train_label=data['label']


input=tf.placeholder(tf.float32,shape=[None,784])
t_image=tf.reshape(input,[-1,28,28,1])
label=tf.placeholder(tf.uint8,shape=[None])
label_onehot=tf.one_hot(label,10,1,0)

conv1=tf.layers.conv2d(t_image,64,(3,3),(1,1),padding='SAME')
conv2=tf.layers.conv2d(conv1,128,(3,3),(1,1),padding='SAME')
conv3=tf.layers.conv2d(conv2,64,(3,3),(1,1),padding='SAME')
flat=tf.layers.Flatten()(conv3)
dense=tf.layers.dense(flat,1024)
logits=tf.layers.dense(dense,10)
#不要用relu 会使output都收敛为0
#logits=tf.nn.relu(dense)
output=tf.nn.softmax(logits)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=label_onehot))
train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss)
correct_prediction=tf.equal(tf.argmax(output,1),tf.argmax(label_onehot,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
with tf.Session() as sess:
    nums=int(np.shape(train_data)[0]/batch_size)
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for step in range(nums):
            acc,_,out,ll=sess.run([accuracy,train_step,output,label_onehot],feed_dict={input:train_data[step*batch_size:(step+1)*batch_size],label:train_label[step*batch_size:(step+1)*batch_size]})
            print('[epoch %s,step%s]:%s'%(epoch,step,acc))



