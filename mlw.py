#!/usr/bin/env python3

import tensorflow as tf

def conv(input, height, width, pre_channel_cnt, this_channel_cnt, stripe, name):
    '''
    卷积层
    '''
    with tf.name_scope(str(name)) as scope:
        kernel = tf.Variable(tf.truncated_normal([height,width,pre_channel_cnt,this_channel_cnt], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(input, kernel, [1,stripe,stripe,1], padding= 'SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[this_channel_cnt], dtype=tf.float32), trainable = True, name='bias')
        product = tf.nn.bias_add(conv, biases)
        return tf.nn.relu(product, name=scope)

def pool(input, fsizeh, fsizew, stride, name):
    '''
    池化层
    '''
    with tf.name_scope(str(name)) as scope:
        return tf.nn.max_pool(input, ksize=[1,fsizeh,fsizew,1], strides=[1,stride,stride,1], padding='VALID', name='pool')

def full_connect(input, pre_nodes, nodes, name, dropout_rate=0, regularizer=None):
    '''
    全连接层
    '''
    with tf.name_scope(str(name)) as scope:
        w = tf.Variable(tf.truncated_normal([pre_nodes, nodes], stddev=0.1))
        b = tf.Variable(tf.zeros(nodes))
        fc_before_dropout = tf.nn.relu(tf.matmul(input,w) + b)
        
        # 对于训练，增加dropout
        if dropout_rate == 0:
            # 不进行dropout：发生在非training或者不打算dropout的情况
            fc = fc_before_dropout
        else:
            fc = tf.nn.dropout(fc_before_dropout, dropout_rate)

        # 正则化
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(w))

    return fc

def softmax(input, pre_nodes, type_cnt, name, regularizer=None):
    '''
    softmax层
    '''
    with tf.name_scope(str(name)) as scope:
        w = tf.Variable(tf.truncated_normal([pre_nodes, type_cnt], stddev=0.1))
        b = tf.Variable(tf.zeros(type_cnt))
        output = tf.nn.softmax(tf.matmul(input,w) + b)

        if regularizer != None:
            tf.add_to_collection('losses',regularizer(w))

    return output