# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 22:35:29 2018
np
@author: rohit
"""
#%%import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#%%import the input data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot= True)

np.random.seed(20180112)
tf.set_random_seed(20180112)
#%%designing the network
num_filters1 = 32
x = tf.placeholder(tf.float32,[None,784])
x_image = tf.reshape(x,[-1,28,28,1])

##firstlayer
#conv with 3x3 filter
W_conv1 = tf.Variable(tf.truncated_normal([3,3,1,num_filters1], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters1]))
h_conv1 = tf.nn.conv2d(x_image, W_conv1,strides=[1,1,1,1],padding='SAME') + b_conv1
#o/p to i/p of another 3x3 conv
W_conv1_1 = tf.Variable(tf.truncated_normal([3,3,num_filters1,num_filters1], stddev=0.1))
b_conv1_1 = tf.Variable(tf.constant(0.1, shape=[num_filters1]))
h_conv1_1 = tf.nn.conv2d(h_conv1,W_conv1_1,strides=[1,1,1,1],padding='SAME')
h_conv1_cutoff = tf.nn.relu(h_conv1_1 + b_conv1_1)
h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

##second layer
num_filters2 = 64
#conv with 3x3 kernels 
W_conv2 = tf.Variable(tf.truncated_normal([3,3,num_filters1,num_filters2],stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]))
h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1],padding='SAME') + b_conv2
#o/p to i/p of another 3x3 conv
W_conv2_1 = tf.Variable(tf.truncated_normal([3,3,num_filters2,num_filters2], stddev=0.1))
b_conv2_1 = tf.Variable(tf.constant(0.1, shape=[num_filters2]))
h_conv2_1 = tf.nn.conv2d(h_conv2, W_conv2_1,strides=[1,1,1,1],padding='SAME')
h_conv2_cutoff = tf.nn.relu(h_conv2_1 + b_conv2_1)
h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
#%%fully connected 
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*num_filters2])
num_units1 = 7*7*num_filters2
num_units2 = 1024

w1 = tf.Variable(tf.truncated_normal([num_units1,num_units2]))
b1= tf.Variable(tf.constant(0.1, shape=[num_units2]))
hidden1 = tf.nn.relu(tf.matmul(h_pool2_flat, w1)+b1)

keep_prob = tf.placeholder(tf.float32)
hidden1_dropout = tf.nn.dropout(hidden1, keep_prob)

w0 = tf.Variable(tf.zeros([num_units2,10]))
b0 = tf.Variable(tf.zeros([10]))
p = tf.nn.softmax(tf.matmul(hidden1_dropout,w0)+b0)
#%%setup for training
t = tf.placeholder(tf.float32,[None,10])
loss = -tf.reduce_sum(t*tf.log(p))
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(p,1), tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#%%session setup
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
#%%training the network
i = 0 
for _ in range(1000):
    i+=1
    batch_xs, batch_ts = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict ={x: batch_xs, t:batch_ts, keep_prob:0.5})
    if i%500 == 0:
        loss_vals , acc_vals =[] , []
        for c in range(4):
            start = int(len(mnist.test.labels)/4*c)
            end = int(len(mnist.test.labels)/4*(c+1))
            loss_val, acc_val = sess.run([loss,accuracy],feed_dict={x:mnist.test.images[start:end], t:mnist.test.labels[start:end],keep_prob:1.0})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
        loss_val=np.sum(loss_vals)
        acc_val =np.mean(acc_vals)
        print('Step: %d; Loss: %f; Accuracy: %f' %(i,loss_val,acc_val))
#%%Display
images, labels = mnist.test.images, mnist.test.labels
p_val = sess.run(p, feed_dict={x:images, t: labels, keep_prob:1.0}) 
fig = plt.figure(figsize=(8,15))
for i in range(10):
    c = 1
    for(image, label, pred) in zip (mnist.test.images, mnist.test.labels, p_val):
        prediction, actual = np.argmax(pred), np.argmax(label)
        if prediction != i:
            continue
        if (c<4 and i == actual) or (c>=4 and i!= actual):
            subplot = fig.add_subplot(10,6,i*6+c)
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.set_title('%d / %d' %(prediction, actual))
            subplot.imshow(image.reshape((28,28)), vmin=0, vmax=1, cmap=plt.cm.gray_r, interpolation = 'nearest')
            c +=1
            plt.show()
            ##plt.pause(1)
            if c>6:
                break
