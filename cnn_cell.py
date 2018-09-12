# coding=utf-8

import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

xs = tf.placeholder(tf.float32, [None, 600])  # 输入图片的大小，28x28=784
ys = tf.placeholder(tf.float32, [None, 1])  # 输出0-9共10个数字
keep_prob = tf.placeholder(tf.float32)  # 用于接收dropout操作的值，dropout为了防止过拟合
# -1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量，
# 因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3
x_image = tf.reshape(xs, [-1, 20, 30, 1])

'''计算准确度函数'''
def compute_accuracy(xs,ys,X,y,keep_prob,sess,prediction):
    y_pre = sess.run(prediction,feed_dict={xs:X,keep_prob:1.0})       # 预测，这里的keep_prob是dropout时用的，防止过拟合，feed_dict是赋值
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(y,1))  #tf.argmax 给出某个tensor对象在某一维上的其数据最大值所在的索引值,即为对应的数字，tf.equal 来检测我们的预测是否真实标签匹配
                                                                      #equal判断两矩阵对应位置元素是否相等，相等则为1，返回矩阵形式与前一个相同
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 平均值即为准确度，cast是类型转换
    result = sess.run(accuracy,feed_dict={xs:X,ys:y,keep_prob:1.0})
    return result



def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)  # 使用truncated_normal进行初始化，生成正态分布，stddev表示标准差，shape表示生成张量的维度
    return tf.Variable(inital)

def bias_variable(shape):
    inital = tf.constant(0.1,shape=shape)  # 偏置定义为常量
    return tf.Variable(inital)

def conv2d(x,W):#x是图片的所有参数，W是此卷积层的权重
    # strides[0]和strides[3]的两个1是默认值，中间两个1代表padding时在x方向运动1步，y方向运动1步
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    # 池化的核函数大小为2x2，因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]，x是池化输入
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1], padding='SAME')

def get_cnn_cell():
    '''第一层卷积，池化'''
    W_conv1 = weight_variable([5, 5, 1, 32])  # 卷积核定义为5x5,1是输入的通道数目，32是输出的通道数目
    b_conv1 = bias_variable([32])  # 每个输出通道对应一个偏置
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 卷积运算，并使用ReLu激活函数激活

    batch_mean1, batch_var1 = tf.nn.moments(h_conv1, [0, 1, 2], keep_dims=True) #计算统计矩，mean 是一阶矩即均值，variance 则是二阶中心矩即方差，axes=[0]表示按列计算
    shift1 = tf.Variable(tf.zeros([32]))
    scale1 = tf.Variable(tf.ones([32]))
    epsilon1 = 1e-3
    BN_out1 = tf.nn.batch_normalization(h_conv1, batch_mean1, batch_var1, shift1, scale1, epsilon1)
    print(BN_out1)
    relu_BN_maps1 = tf.nn.relu(BN_out1)
    h_pool1 = max_pool_2x2(relu_BN_maps1)


    '''第二层卷积，池化'''
    W_conv2 = weight_variable([5,5,32,64]) # 卷积核还是5x5,32个输入通道，64个输出通道
    b_conv2 = bias_variable([64])          # 与输出通道一致
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)

    # BN归一化层+激活层+池化
    batch_mean2, batch_var2 = tf.nn.moments(h_conv2, [0, 1, 2], keep_dims=True)
    shift2 = tf.Variable(tf.zeros([64]))
    scale2 = tf.Variable(tf.ones([64]))
    epsilon2 = 1e-3
    BN_out2 = tf.nn.batch_normalization(h_conv2, batch_mean2, batch_var2, shift2, scale2, epsilon2)
    print(BN_out2)
    relu_BN_maps2 = tf.nn.relu(BN_out2)
    h_pool2 = max_pool_2x2(relu_BN_maps2)

    '''第三层卷积，池化'''
    W_conv3 = weight_variable([5, 5, 64, 64])  # 卷积核还是5x5,32个输入通道，64个输出通道
    b_conv3 = bias_variable([64])  # 与输出通道一致
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    # BN归一化层+激活层+池化
    batch_mean3, batch_var3 = tf.nn.moments(h_conv3, [0, 1, 2], keep_dims=True)
    shift3 = tf.Variable(tf.zeros([64]))
    scale3 = tf.Variable(tf.ones([64]))
    epsilon3 = 1e-3
    BN_out3 = tf.nn.batch_normalization(h_conv3, batch_mean3, batch_var3, shift3, scale3, epsilon3)
    print(BN_out3)
    relu_BN_maps3 = tf.nn.relu(BN_out3)
    h_pool3 = max_pool_2x2(relu_BN_maps3)


    '''全连接层'''
    h_pool3_flat = tf.reshape(h_pool3, [-1, 3 * 4 * 64])  # 将最后操作的数据展开
    W_fc1 = weight_variable([3 * 4 * 64, 1024])  # 下面就是定义一般神经网络的操作了，继续扩大为1024
    b_fc1 = bias_variable([1024])  # 对应的偏置
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)  # 运算、激活（这里不是卷积运算了，就是对应相乘

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # dropout操作

    '''最后一层全连接'''
    W_fc2 = weight_variable([1024,10])                # 最后一层权重初始化
    # b_fc2 = bias_variable([10])                       # 对应偏置
    b_fc2 = bias_variable([10])  # 对应偏置

    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)  # 使用softmax分类器
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))  # 交叉熵损失函数来定义cost function
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 调用梯度下降

    return prediction, train_step