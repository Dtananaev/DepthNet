#
# Author: Denis Tananaev
# File: losses.py
# Date: 9.02.2017
# Description: loss functions for neural networks
#
#include libs
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function




import os
import re
import sys
import tarfile
import math 
import tensorflow as tf 

#Benjamin modules
sys.path.insert(0,'../external/tfspecialops/python')
from tfspecialops import tfspecialops as ops

LOSSES_COLLECTION = 'losses'


def L2loss(output, gt,batch_size,scope=None):
    with tf.name_scope(scope, 'L2loss', [output, gt]):
        zero=tf.zeros_like(gt)
        mask = tf.not_equal(gt, zero)
        mask=tf.cast(mask,tf.float32)
        # compute L2 loss
        diff=tf.multiply(mask,tf.subtract(output,gt))#find difference
        L2loss=tf.reduce_sum(tf.square(diff))/(2*batch_size)
        tf.add_to_collection(LOSSES_COLLECTION, L2loss)
        return L2loss
        

def convertNHWC2NCHW(data):
       #[N,H,W,C] = [N,C,H,W]
        out = tf.transpose(data, [0, 3, 1, 2])
        return out
    
def convertNCHW2NHWC(data):
        #[N,C,H,W]=[N,H,W,C] 
        out = tf.transpose(data, [0, 2, 3, 1])
        return out
    
def scinv_gradloss(output, gt,batch_size,scope=None):
    
    with tf.name_scope(scope, 'scinv_gradloss', [output, gt]):
        #convert from NHWC to NCHW
        output=convertNHWC2NCHW(output)
        gt=convertNHWC2NCHW(gt)        
        #pad the tensor
        paddings=[[0,0],[1,0],[2,4],[3,4]]# the 3d and 4th dimention pad with 4 zero from each side
        output=tf.pad(output,paddings,'CONSTANT')
        gt=tf.pad(gt,paddings,'CONSTANT')
        # compute mask and make zero areas NaN in order to remove them later
        zero=tf.zeros_like(gt)
        mask = tf.not_equal(gt, zero)
        mask=tf.cast(mask,tf.float32)
        #mask=tf.div(mask,mask)# divide 0/0 gives us NaN values
        #output_nan= tf.div(output,mask)
        #gt_nan=tf.div(gt,mask)
        # compute scale invariant grad loss                
        grad_output = ops.scale_invariant_gradient(input=output, deltas=[1,2,4], weights=[1,0.5,0.25], epsilon=0.001)
        grad_gt = ops.scale_invariant_gradient(input=gt, deltas=[1,2,4], weights=[1,0.5,0.25], epsilon=0.001)
        diff = ops.replace_nonfinite(grad_output-grad_gt)
        #diff=tf.subtract(grad_output,grad_gt)
        #apply mask
        #mask_out=tf.concat(0, [mask,mask])# the grad has 2 time more matrices in tensor because of grad by x and by y
        #diff=tf.multiply(mask_out,diff)
        #remove NaN values
        #diff=tf.select(tf.is_nan(diff),tf.zeros_like(diff),diff)

        gradLoss=tf.reduce_sum(tf.square(diff))/(2*batch_size)
        tf.add_to_collection(LOSSES_COLLECTION, gradLoss)
        return gradLoss
