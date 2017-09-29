#!/usr/bin/env python3
# _*_ coding:utf-8 _*_


'''
Hello, you can use this module to load images
'''

import os
import tensorflow as tf
from PIL import Image
import imghdr
import numpy as np
import pdb

def p(*args):
    '''
    moduel print
    '''
    print(__name__ + ': ', end='')
    for x in args:
        print(x, end=' ')
    print('')

def load_img(path, do_resize = False, height=0, width=0, ):
    '''
    加载指定高度与宽度的图片。默认JPEG，输出是len为图片个数的 np.ndarray
    '''
    if not os.path.isdir(path):
        p('not a directory')
        return 

    size_fixed = True           # 输出长宽是否是固定值
    if height == width == 0:
        size_fixed = False;

    path_list = os.listdir(path)
    path_list.sort()
    img_name_list = [];     # 文件名
    img_full_name_list = [];     # 文件名
    img_data_list = [];     # 文件数据

    # 遍历路径
    for fpath in path_list:

        if not os.path.isfile(path + '/' + fpath):
            p((path + '/' + fpath), 'is not a file')
            continue
        
        if imghdr.what(path + '/' + fpath) != 'jpeg':
            p((path + '/' + fpath), 'is not a jpeg image')
            continue
               
        try:
            with Image.open(path + '/' + fpath) as img:

                if size_fixed and img.size != (width, height):
                    p('size not match -', fpath, img.size)
                    continue
                
                reader = tf.WholeFileReader()
                key,value = reader.read(tf.train.string_input_producer([path + '/' + fpath]))
                [path + '/' + fpath]
                image = tf.image.decode_jpeg(value)
                with tf.Session() as sess:
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    raw_data = sess.run(image)
                    coord.request_stop()
                    coord.join(threads)
                    image_uint8 = tf.image.convert_image_dtype(image,dtype=tf.uint8)

                img_name_list.append(fpath)
                img_data_list.append(raw_data)

                print(fpath, 'shape is', raw_data.shape)


        except Exception as excp:
            p('fpath open failed. ', excp)
   
    print(img_name_list)
    for it in img_full_name_list:
        print(it)
    print('Totally Count:', len(img_name_list))

    return img_name_list, img_data_list


if __name__ == '__main__':

    print('in TESING MODE...')
    pdb.set_trace()
    load_img("../../../dataset/VOCdevkit/pix2/c1", False, 375, 500)

