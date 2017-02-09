#
# Author: Denis Tananaev
# File: makeTFrecords.py
# Date:9.02.2017
# Description:  tool for the tfrecords convertion of the SUN3D dataset
#

import numpy as np
import skimage.io as io
import scipy.misc
import tensorflow as tf


def centered_crop(image,new_w,new_h):
    '''Make centered crop of the image'''
    height = image.shape[0]
    width = image.shape[1]
    left = (width - new_w)/2
    top = (height - new_h)/2
    right = (width + new_w)/2
    bottom = (height + new_h)/2
    return image[top:bottom,left:right]


def resizer_image(image):
    '''Resize images by using bilinear interpolation'''
    croped_image=centered_crop(image,550,450)
    result=scipy.misc.imresize(croped_image, (192,256), interp='bilinear', mode=None)
    return result

def resizer_depth(depth):
    '''Resize depth by using nearest neighbour method '''
    croped_image=centered_crop(depth,550,450)
    result=scipy.misc.imresize(croped_image, (192,256), interp='nearest', mode=None)
    return result



def read_file(textfile):
  '''Read txt file and output array of strings line by line '''
  with open(textfile) as f:
    result = f.read().splitlines()
  return result

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def make_tfrecords(tfrecords_filename,filename_pairs):
    '''Convert pairs of (image, depth) tuple to the tfrecords format'''
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for img_path, depth_path in filename_pairs:
    
        img = np.array(io.imread(img_path))
        depth = np.array(io.imread(depth_path))
    
        # The reason to store image sizes was demonstrated
        # in the previous example -- we have to know sizes
        # of images to later read raw serialized string,
        # convert to 1d array and convert to respective
        # shape that image used to have.
        img=resizer_image(img)
        depth=resizer_depth(depth)
        img_raw = img.tostring()
        depth_raw = depth.tostring()
        height = img.shape[0]
        width = img.shape[1]
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'depth_raw': _bytes_feature(depth_raw)}))
    
        writer.write(example.SerializeToString())

    writer.close()


def createPairs(train_im,train_d):
    '''Create array of tuples (image,depth) '''
    #read the list of pathes to jpg data from txt
    input_list=read_file(train_im)
    #read the list of pathes to png data from txt
    output_list=read_file(train_d)
    result=[]
    for i in range(0,len(input_list)):
        temp=(input_list[i],output_list[i])
        result.append(temp)
    return result
