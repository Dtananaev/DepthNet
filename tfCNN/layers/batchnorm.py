#
# File: batchnorm.py
# Date:27.01.2017
# Author: Denis Tananaev
#
#
#include libs
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import os
import re
import sys
import tarfile
import math 
import tensorflow as tf
import summary as sm
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm



UPDATE_OPS_COLLECTION = '_update_ops_'


def batch_norm_layer(x,train_phase,scope_bn,reuse=None):
    with tf.variable_scope(scope_bn, 'BatchNorm', [x], reuse=reuse):
        z = batch_norm(x, decay=0.999, center=True, scale=True,is_training=train_phase,
        reuse=reuse,
        trainable=True,
        scope=scope_bn,updates_collections=UPDATE_OPS_COLLECTION)

        return z
    
    


