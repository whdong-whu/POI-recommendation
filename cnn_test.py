# coding=utf-8
import tensorflow as tf
import cnn_cell as cnn
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
# from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':


    prediction, train_step = cnn.get_cnn_cell()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_xs = np.zeros((2859400, 600), dtype=np.float32)  # 先创建一个 3x3的全零方阵A，并且数据的类型设置为float浮点型

    f = open('2.txt')  # 打开数据文件文件
    lines = f.readlines()  # 把全部数据文件读到一个列表lines中
    A_row = 0  # 表示矩阵的行，从0行开始
    for line in lines:  # 把lines中的数据逐行读取出来
        dList= line.strip('\n').split(' ')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        # print(len(dList))
        batch_xs[A_row:] = dList[0:600]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
        A_row += 1  # 然后方阵A的下一行接着读
        # print(line)
    # print(batch_xs.dtype,batch_xs.shape) #float64 (70, 784)
    # b = tf.cast(batch_xs, tf.float32)
    # print(b.dtype)


    batch_ys= np.zeros((2859400, 1), dtype=np.float32)  # 先创建一个 3x3的全零方阵A，并且数据的类型设置为float浮点型
    f = open('y_train.txt')  # 打开数据文件文件
    lines = f.readlines()  # 把全部数据文件读到一个列表lines中
    A_row = 0  # 表示矩阵的行，从0行开始
    for line in lines:  # 把lines中的数据逐行读取出来
        dlist = line.strip('\n').split(' ')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        batch_ys[A_row:] = dlist[0:1]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
        A_row += 1
        m_saver = tf.train.Saver()

        sess.run(train_step, feed_dict={cnn.xs: batch_xs, cnn.ys: batch_ys, cnn.keep_prob: 0.5})
        m_saver.save(sess, '/home/machine/workspace/mrd/poi_slp', global_step=1)

    #
    #
    # x_test = np.zeros((163700650, 600), dtype=np.float32)  # 先创建一个 3x3的全零方阵A，并且数据的类型设置为float浮点型
    # f = open('x_test.txt')  # 打开数据文件文件
    # lines = f.readlines()  # 把全部数据文件读到一个列表lines中
    # A_row = 0  # 表示矩阵的行，从0行开始
    # for line in lines:  # 把lines中的数据逐行读取出来
    #     list = line.strip('\n').split(' ')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
    #     x_test[A_row:] = list[0:600]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
    #     A_row += 1
    #
    # y_test = np.zeros((163700650, 1), dtype=np.float32)  # 先创建一个 3x3的全零方阵A，并且数据的类型设置为float浮点型
    # f = open('y_test.txt')  # 打开数据文件文件
    # lines = f.readlines()  # 把全部数据文件读到一个列表lines中
    # A_row = 0  # 表示矩阵的行，从0行开始
    # for line in lines:  # 把lines中的数据逐行读取出来
    #     list = line.strip('\n').split(' ')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
    #     y_test[A_row:] = list[0:1]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
    #     A_row += 1
    # accuracy = cnn.compute_accuracy(cnn.xs, cnn.ys, x_test, y_test, cnn.keep_prob, sess, prediction)
    # print(accuracy)
