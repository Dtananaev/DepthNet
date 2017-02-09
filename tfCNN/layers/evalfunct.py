#
# Author: Denis Tananaev
# File: evalfunct.py
# Date: 9.02.2017
# Description: evaluation functions for neural networks
#

#include libs
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from six.moves import xrange
import os
import re
import sys
import tarfile
import math 
import tensorflow as tf


def scinv(result,gt):
    #don't eval on zero points in depth images
    zero=tf.zeros_like(gt)
    mask = tf.not_equal(gt, zero)
    mask=tf.cast(mask,tf.float32)    
    n=tf.reduce_sum(mask)# number of elements for evaluation
    
    d=tf.subtract(tf.log(result),tf.log(gt))
    d=tf.multiply(mask,d)
    dsq=tf.reduce_sum(tf.square(d))
    
    error=tf.sqrt( (1/n)*dsq - (1/(n*n))* tf.square(tf.reduce_sum(d)))   
    return error

def L1rel(result,gt):
    #don't eval on zero points in depth images
    zero=tf.zeros_like(gt)
    mask = tf.not_equal(gt, zero)
    mask=tf.cast(mask,tf.float32)    
    n=tf.reduce_sum(mask)# number of elements for evaluation
    
    error=(1/n)*tf.reduce_sum(tf.multiply(mask,tf.abs(tf.div(tf.subtract(result,gt),gt))))
    return error
    
    
    

def L1inv(result,gt):
    #don't eval on zero points in depth images
    zero=tf.zeros_like(gt)
    mask = tf.not_equal(gt, zero)
    mask=tf.cast(mask,tf.float32)    
    n=tf.reduce_sum(mask)# number of elements for evaluation
    one=tf.ones_like(gt)
    invgt=tf.div(one,gt)
    invresult=tf.div(one,result)
    
    error=(1/n)*tf.reduce_sum(tf.multiply(mask,tf.abs(tf.subtract(invgt,invresult))))
    return error