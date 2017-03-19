# -*- coding:utf-8 -*-

## 导入示例数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)      # y为预测分布

# 定义loss function: cross_entropy
y_ = tf.placeholder(tf.float32, [None, 10]) # y_为真实分布
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 定义优化算法为SGD 学习速率0.5 优化目标是交叉熵
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 执行全局初始化器
tf.global_variables_initializer().run()

# mini-batch 迭代训练
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100) # 每次随机抽取100条组成mini-batch
    train_step.run({x: batch_xs, y_: batch_ys})      # feed给placeholder

# 验证准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x:mnist.test.images, y_: mnist.test.labels}))







