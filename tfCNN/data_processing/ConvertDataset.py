#
# Author: Denis Tananaev
# File: ConvertDataset.py
# Date:9.02.2017
# Description: parser tool for the files for SUN3D dataset
#

#include libs
import numpy as np
import glob
import os, sys
import re
import tensorflow as tf
import tools.parser as parser
import tools.makeTFrecords as mtf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset_dir', 
                           '/misc/lmbraid11/tananaed/DepthNet/SUN3D/Dataset/',
                           """Path to sun3d dataset """)

tf.app.flags.DEFINE_string('lists_dir', './lists/',
                           """Directory where to write lists of pathes to the data """)

tf.app.flags.DEFINE_string('tfrecords_dir', './dataset/',
                           """Directory where to save tfrecords """)

tf.app.flags.DEFINE_string('convert', 'yes',
                           """if yes convert dataset to tf recors """)
def convert():
    print('dataset folder:', FLAGS.dataset_dir)
    print('lists output directory:',FLAGS.lists_dir)
    print('tfrecords output directory:',FLAGS.tfrecords_dir)
    print('convert dataset to tfrecords:',FLAGS.convert) 
    
    print('start parcing the dataset folders in order to create the lists of datasamples')
    parser.makeLists(Path_to_dataset_folder=FLAGS.dataset_dir,
              list_trainset='./list_train.txt',
              list_testset='./list_test.txt',
              output_tr_depth=FLAGS.lists_dir+"depth_train.txt",
              output_tr_image=FLAGS.lists_dir+"image_train.txt",
              output_test_image=FLAGS.lists_dir+"image_test.txt",
              output_test_depth=FLAGS.lists_dir+"depth_test.txt")
    print('complete.')
    
    if FLAGS.convert=='yes':
        tfrecords_filename_train =FLAGS.tfrecords_dir+ 'train_sun3d.tfrecords'
        tfrecords_filename_test = FLAGS.tfrecords_dir+'test_sun3d.tfrecords'

        train_im_list=FLAGS.lists_dir+'image_train.txt'
        train_d_list=FLAGS.lists_dir+'depth_train.txt'
        test_im_list=FLAGS.lists_dir+'image_test.txt'
        test_d_list=FLAGS.lists_dir+'depth_test.txt'
        

        filename_pairs_train=mtf.createPairs(train_im_list,train_d_list)
        filename_pairs_test=mtf.createPairs(test_im_list,test_d_list)
        print('Start convertion of the train set to tfrecors...')
        mtf.make_tfrecords(tfrecords_filename_train,filename_pairs_train)
        print('complete.')
        print('Start convertion of the test set to tfrecors...')
        mtf.make_tfrecords(tfrecords_filename_test,filename_pairs_test)
        print('complete.')



def main(argv=None):
  # lists_dir
  if tf.gfile.Exists(FLAGS.lists_dir):
    tf.gfile.DeleteRecursively(FLAGS.lists_dir)
  tf.gfile.MakeDirs(FLAGS.lists_dir)
  # tfrecords_dir
  if tf.gfile.Exists(FLAGS.tfrecords_dir):
    tf.gfile.DeleteRecursively(FLAGS.tfrecords_dir)
  tf.gfile.MakeDirs(FLAGS.tfrecords_dir)
  convert()
  
  
if __name__ == '__main__':
  tf.app.run()  
    
    
    

