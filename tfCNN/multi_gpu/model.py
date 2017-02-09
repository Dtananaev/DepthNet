#
# File: model.py
# Date:21.01.2017
# Author: Denis Tananaev
# 
#


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.insert(0, '../layers/')
import os
import re
import tarfile
import math 
import tensorflow as tf
#layers
import summary as sm
import conv as cnv
import activations as act
import deconv as dcnv
import batchnorm as bn
FLAGS = tf.app.flags.FLAGS
import losses as lss

import data
import param
#parameters of the datasets
IMAGE_SIZE_W=param.IMAGE_SIZE_W
IMAGE_SIZE_H=param.IMAGE_SIZE_H
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=param.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = param.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
INVERSE_DEPTH=param.INVERSE_DEPTH
FLOAT16=param.FLOAT16
#parameters of the data uploading
BATCH=param.BATCH_SIZE
NUM_GPU=param.NUM_GPU
# Constants describing the training process.
MOVING_AVERAGE_DECAY = param.MOVING_AVERAGE_DECAY   # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = param.NUM_EPOCHS_PER_DECAY     # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = param.LEARNING_RATE_DECAY_FACTOR  # Learning rate decay factor.
INITIAL_LEARNING_RATE = param.INITIAL_LEARNING_RATE     # Initial learning rate.
WEIGHT_DECAY=param.WEIGHT_DECAY


def inputs(eval_data=False):     
  images, depths = data.read_sun3d(eval_data=eval_data)
  return images, depths


def inference(images, phase_train,scope=''):
    BATCH_SIZE=int(BATCH/NUM_GPU)
    with tf.name_scope(scope, [images]):
        #Conv11-64p1
        conv0=cnv.conv(images,'conv0',[11, 11, 3, 32],stride=[1,1,1, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm0=bn.batch_norm_layer(conv0,train_phase=phase_train,scope_bn='BN0')
        relu0=act.ReLU(bnorm0,'ReLU0') 
        #SKIP CONNECTION 0
        #Conv9-128s2
        conv1=cnv.conv(relu0,'conv1',[9, 9, 32, 64],stride=[1,2,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm1=bn.batch_norm_layer(conv1,train_phase=phase_train,scope_bn='BN1')
        relu1=act.ReLU(bnorm1,'ReLU1')
        #Conv3-128p1  
        conv2=cnv.conv(relu1,'conv2',[3, 3, 64, 128],wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        bnorm2=bn.batch_norm_layer(conv2,train_phase=phase_train,scope_bn='BN2')
        relu2=act.ReLU(bnorm2,'ReLU2')
        #Conv3-128p1    
        conv3=cnv.conv(relu2,'conv3',[3, 3, 128, 128],wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        bnorm3=bn.batch_norm_layer(conv3,train_phase=phase_train,scope_bn='BN3')
        relu3=act.ReLU(bnorm3,'ReLU3')
        #SKIP CONNEgradLossCTION 1
        #Conv7-256s2
        conv4=cnv.conv(relu3,'conv4',[7, 7, 128, 256],stride=[1,2,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm4=bn.batch_norm_layer(conv4,train_phase=phase_train,scope_bn='BN4')
        relu4=act.ReLU(bnorm4,'ReLU4')
        #Conv3-256p1 
        conv5=cnv.conv(relu4,'conv5',[3, 3, 256, 256],wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        bnorm5=bn.batch_norm_layer(conv5,train_phase=phase_train,scope_bn='BN5')
        relu5=act.ReLU(bnorm5,'ReLU5')
        #Conv3-256p1    
        conv6=cnv.conv(relu5,'conv6',[3, 3, 256, 256],wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        bnorm6=bn.batch_norm_layer(conv6,train_phase=phase_train,scope_bn='BN6')
        relu6=act.ReLU(bnorm6,'ReLU6')
        #SKIP CONNECTION 2 
        #Conv5-512s2
        conv7_1=cnv.conv(relu6,'conv7_1',[5, 1, 256, 512],stride=[1,2,1, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        conv7_2=cnv.conv(conv7_1,'conv7_2',[1, 5, 512, 512],stride=[1,1,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm7=bn.batch_norm_layer(conv7_2,train_phase=phase_train,scope_bn='BN7')
        relu7=act.ReLU(bnorm7,'ReLU7')
        #Conv3-512p1
        conv8_1=cnv.conv(relu7,'conv8_1',[3, 1, 512, 512],wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        conv8_2=cnv.conv(conv8_1,'conv8_2',[1, 3, 512, 512],wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        bnorm8=bn.batch_norm_layer(conv8_2,train_phase=phase_train,scope_bn='BN8')
        relu8=act.ReLU(bnorm8,'ReLU8')
        #Conv3-512p1    
        conv9_1=cnv.conv(relu8,'conv9_1',[1, 3, 512, 512],wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        conv9_2=cnv.conv(conv9_1,'conv9_2',[3, 1, 512, 512],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm9=bn.batch_norm_layer(conv9_2,train_phase=phase_train,scope_bn='BN9')
        relu9=act.ReLU(bnorm9,'ReLU9')  
        #SKIP CONNECTION 3  
        #Conv3-1024s2
        conv10_1=cnv.conv(relu9,'conv10_1',[3, 1, 512, 1024],stride=[1,2,1, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        conv10_2=cnv.conv(conv10_1,'conv10_2',[1, 3, 1024, 1024],stride=[1,1,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        bnorm10=bn.batch_norm_layer(conv10_2,train_phase=phase_train,scope_bn='BN10')
        relu10=act.ReLU(bnorm10,'ReLU10') 
        #Conv3-1024p1
        conv11_1=cnv.conv(relu10,'conv1UPDATE_OPS_COLLECTION1_1',[1, 3, 1024, 1024],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        conv11_2=cnv.conv(conv11_1,'conv11_2',[3, 1, 1024, 1024],wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        bnorm11=bn.batch_norm_layer(conv11_2,train_phase=phase_train,scope_bn='BN11')
        relu11=act.ReLU(bnorm11,'ReLU11') 
  
        #GO UP  
        deconv1=dcnv.deconv(relu11,[BATCH_SIZE,int(IMAGE_SIZE_H/8),int(IMAGE_SIZE_W/8),512],'deconv1',[4, 4, 512, 1024],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        dbnorm1=bn.batch_norm_layer(deconv1,train_phase=phase_train,scope_bn='dBN1')
        drelu1=act.ReLU(dbnorm1+relu9,'dReLU1')
  
        conv12_1=cnv.conv(drelu1,'conv12_1',[3, 1, 512, 512],wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        conv12_2=cnv.conv(conv12_1,'conv12_2',[1, 3, 512, 512],wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        bnorm12=bn.batch_norm_layer(conv12_2,train_phase=phase_train,scope_bn='BN12')
        relu12=act.ReLU(bnorm12,'ReLU12')
  
        deconv2=dcnv.deconv(relu12,[BATCH_SIZE,int(IMAGE_SIZE_H/4),int(IMAGE_SIZE_W/4),256],'deconv2',[4, 4, 256, 512],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        dbnorm2=bn.batch_norm_layer(deconv2,train_phase=phase_train,scope_bn='dBN2')
        drelu2=act.ReLU(dbnorm2+relu6,'dReLU2')
  
        conv13=cnv.conv(drelu2,'conv13',[3, 3, 256, 256],wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        bnorm13=bn.batch_norm_layer(conv13,train_phase=phase_train,scope_bn='BN13')
        relu13=act.ReLU(bnorm13,'ReLU13')
  
        deconv3=dcnv.deconv(relu13,[BATCH_SIZE,int(IMAGE_SIZE_H/2),int(IMAGE_SIZE_W/2),128],'deconv3',[4, 4, 128, 256],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        dbnorm3=bn.batch_norm_layer(deconv3,train_phase=phase_train,scope_bn='dBN3')
        drelu3=act.ReLU(dbnorm3+relu3,'dReLU3')
  
        conv14=cnv.conv(drelu3,'conv14',[3, 3, 128, 128],wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        bnorm14=bn.batch_norm_layer(conv14,train_phase=phase_train,scope_bn='BN14')
        relu14=act.ReLU(bnorm14,'ReLU14') 
  
        deconv4=dcnv.deconv(relu14,[BATCH_SIZE,int(IMAGE_SIZE_H),int(IMAGE_SIZE_W),32],'deconv4',[4, 4, 32, 128],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16)
        dbnorm4=bn.batch_norm_layer(deconv4,train_phase=phase_train,scope_bn='dBN4')
        drelu3=act.ReLU(dbnorm4+relu0,'dReLU4')
  
        conv_last=cnv.conv(drelu3,'conv_last',[3, 3, 32, 32],wd=WEIGHT_DECAY,FLOAT16=FLOAT16) 
        bnorm_last=bn.batch_norm_layer(conv_last,train_phase=phase_train,scope_bn='BNl')
        relu_last=act.ReLU(bnorm_last,'ReLU_last')   
  
        scores=cnv.conv(relu_last,'scores',[3, 3, 32, 1],wd=0,FLOAT16=FLOAT16)
        tf.summary.image('output', scores)  

        return scores



def loss(images,depths,batch_size=None):
    if not batch_size:
      batch_size=BATCH_SIZE
    lss.L2loss(images, depths,batch_size)
    lss.scinv_gradloss(images,depths,batch_size)
    
    



