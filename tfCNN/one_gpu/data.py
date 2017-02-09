#
# File: cnn_data.py
# Date:25.01.2017
# Author: Denis Tananaev
#
#
#include libs


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.insert(0, '../layers/')
from six.moves import xrange
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import param
#datasets
tfrecords_filename_train=param.tfrecords_filename_train
tfrecords_filename_test=param.tfrecords_filename_test
#parameters of the datasets
IMAGE_SIZE_W=param.IMAGE_SIZE_W
IMAGE_SIZE_H=param.IMAGE_SIZE_H
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=param.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = param.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
INVERSE_DEPTH=param.INVERSE_DEPTH
FLOAT16=param.FLOAT16
#parameters of the data uploading
BATCH_SIZE=param.BATCH_SIZE
NUM_PREPROCESS_THREADS=param.NUM_PREPROCESS_THREADS
NUM_READERS=param.NUM_READERS
INPUT_QUEUE_MEMORY_FACTOR=param.INPUT_QUEUE_MEMORY_FACTOR
DATA_SHUFFLE=param.DATA_SHUFFLE
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=param.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
LEARNING_RATE_DECAY_FACTOR=param.LEARNING_RATE_DECAY_FACTOR


def parse_example_proto(example_serialized):
    features = tf.parse_single_example(example_serialized,
        # Defaults are not specified since both keys are required.
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'depth_raw': tf.FixedLenFeature([], tf.string)
            })
    
    return features['image_raw'], features['depth_raw']   
    
def preprocess(image_buffer,depth_buffer):
    image = tf.decode_raw(image_buffer, tf.uint8)
    depth = tf.decode_raw(depth_buffer, tf.uint8)
    image = tf.reshape(image,  tf.pack([IMAGE_SIZE_H, IMAGE_SIZE_W, 3]))
    depth = tf.reshape(depth, tf.pack([IMAGE_SIZE_H, IMAGE_SIZE_W, 1]))
    
    return image,depth


def batch_inputs( batch_size, eval_data, num_preprocess_threads=None,num_readers=None):

    with tf.name_scope('batch_processing'):
        #1 checks the parameters
        if num_preprocess_threads % 4:
            raise ValueError('Please make num_preprocess_threads a multiple '
                       'of 4 (%d % 4 != 0).', num_preprocess_threads)
        if num_readers < 1:
            raise ValueError('Please make num_readers at least 1')
        if num_preprocess_threads < 1:
            raise ValueError('Please make num_preprocess_threads at least 1')
        #2 download dataset
        if eval_data==False:
            filename_queue = tf.train.string_input_producer([tfrecords_filename_train],shuffle=DATA_SHUFFLE,capacity=16)
        else:
            filename_queue = tf.train.string_input_producer([tfrecords_filename_test],shuffle=DATA_SHUFFLE,capacity=16)
            
        #3 allocate the size of the queue of data
        # Approximate number of examples per shard.
        examples_per_shard = 1024
        # Size the random shuffle queue to balance between good global
        # mixing (more examples) and memory use (fewer examples).
        # 1 image uses 256*192*3*4 bytes + 1 depth 256*192*1*4 bytes  = 1MB
        # The default input_queue_memory_factor is 4 implying a shuffling queue
        # size: examples_per_shard * 4 * 1MB = 4GB
        min_queue_examples = examples_per_shard * INPUT_QUEUE_MEMORY_FACTOR
        examples_queue = tf.RandomShuffleQueue(capacity=min_queue_examples + 3 * batch_size,min_after_dequeue=min_queue_examples,dtypes=[tf.string])

        #4 Create multiple readers to populate the queue of examples.
        if num_readers > 1:
            enqueue_ops = []
            for _ in range(num_readers):
                reader = tf.TFRecordReader()
                _, value = reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))

            tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            example_serialized = examples_queue.dequeue()
        else:
            reader = tf.TFRecordReader()
            _, example_serialized = reader.read(filename_queue)
        #5 read serialized data pairs from .tfrecords file         
        images_and_depths = []
        for thread_id in range(num_preprocess_threads):
            # Parse a serialized Example proto to extract the image and metadata.
            image_buffer, depth_buffer = parse_example_proto(example_serialized)
            im,dpth=preprocess(image_buffer,depth_buffer)
            images_and_depths.append([im,dpth])
            
        #6 create batches of images and depths
        images, depths = tf.train.batch_join(images_and_depths,batch_size=batch_size,capacity=2 * num_preprocess_threads * batch_size)

        #7 Reshape images and depths into these desired dimensions
        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, IMAGE_SIZE_H, IMAGE_SIZE_W, 3])
        depths= tf.cast(depths, tf.float32)
        depths = tf.reshape(depths, shape=[batch_size, IMAGE_SIZE_H, IMAGE_SIZE_W, 1])
        #8 rescale depth values from millimeters to meters 
        depths =tf.scalar_mul(0.001,depths)
        #9 Display the training images and depthes in the visualizer.
        tf.summary.image('images', images)
        tf.summary.image('depths', depths)
        
    return images, depths

def read_sun3d(eval_data=False, batch_size=BATCH_SIZE, num_preprocess_threads=NUM_PREPROCESS_THREADS):
  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation.
    with tf.device('/cpu:0'):
        images, depth = batch_inputs(
         batch_size, eval_data=eval_data,
        num_preprocess_threads=num_preprocess_threads,
        num_readers=NUM_READERS)

    return images, depth







