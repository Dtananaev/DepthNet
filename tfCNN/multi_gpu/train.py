#
# File: cnn_train.py
# Date:21.01.2017
# Author: Denis Tananaev
# 
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.insert(0, '../layers/')
import os.path
from datetime import datetime
import time
import re
import tensorflow as tf
import copy
import model
import param
import data
import losses as lss
import batchnorm as bn
import numpy as np

NUM_GPU=param.NUM_GPU
MOVING_AVERAGE_DECAY=param.MOVING_AVERAGE_DECAY
BATCH_SIZE=param.BATCH_SIZE
NUM_EPOCHS_PER_DECAY=param.NUM_EPOCHS_PER_DECAY
INITIAL_LEARNING_RATE=param.INITIAL_LEARNING_RATE
NUM_PREPROCESS_THREADS=param.NUM_PREPROCESS_THREADS
TOWER_NAME=param.TOWER_NAME
LOG_DEVICE_PLACEMENT=param.LOG_DEVICE_PLACEMENT
PRETRAINED_MODEL_CHECKPOINT_PATH=param.PRETRAINED_MODEL_CHECKPOINT_PATH
TRAIN_LOG=param.TRAIN_LOG
NUM_ITER=param.NUM_ITER
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=param.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
LEARNING_RATE_DECAY_FACTOR=param.LEARNING_RATE_DECAY_FACTOR


def _tower_loss(images, depths, scope, reuse_variables=None):
    
  """Calculate the total loss on 
a single tower running the SUN3D model.
  We perform 'batch splitting'. This means that we cut up a batch across
  multiple GPU's. For instance, if the batch size = 32 and num_gpus = 2,
  then each tower will operate on an batch of 16 images.
  """
  #1 Build inference Graph.
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
    scores = model.inference(images,phase_train=True,scope=scope)

  #2 Build the portion of the Graph calculating the losses. 
  # Note: that we will assemble the total_loss using a custom function below.
  split_batch_size = images.get_shape().as_list()[0]
  model.loss(scores, depths, batch_size=split_batch_size)

  #3 Assemble all of the losses for the current tower only.
  losses = tf.get_collection(lss.LOSSES_COLLECTION, scope=scope)

  #4 Calculate the total loss for the current tower.
  regularization_losses=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

  #5 Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, name='avg')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  #6 Attach a scalar summmary to all individual losses and the total loss; do the same for the averaged version of the losses.
  
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on TensorBoard.
    loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(loss_name +' (raw)', l)
    tf.summary.scalar(loss_name, loss_averages.average(l))

  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)
  return total_loss


def _average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lisbnts of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
  """Train on dataset for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    #1 Create a variable to count the number of train() calls.
    # This equals the number of batches processed * NUM_GPU.
    global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)

    #2 Calculate the learning rate schedule.
    num_batches_per_epoch = (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             BATCH_SIZE)
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    #Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    #4 Create an optimizer that performs gradient descent.
    opt = tf.train.AdamOptimizer(lr)

    #5 Get images and labels for ImageNet and split the batch across GPUs.
    assert BATCH_SIZE % NUM_GPU == 0, (
        'Batch size must be divisible by number of GPUs')
    split_batch_size = int(BATCH_SIZE / NUM_GPU)

    #6 Override the number of preprocessing threads to account for the increased number of GPU towers.
    num_preprocess_threads = NUM_PREPROCESS_THREADS * NUM_GPU
    images, depths =  data.read_sun3d(eval_data=False, batch_size=BATCH_SIZE, num_preprocess_threads=num_preprocess_threads)

    input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

     # Split the batch of images and depths for towers.
    images_splits = tf.split(0, NUM_GPU, images)
    depthes_splits = tf.split(0, NUM_GPU, depths)

    # Calculate the gradients for each model tower.
    tower_grads = []
    reuse_variables = None
    for i in range(NUM_GPU):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
          #A) Calculate the loss for one tower of the ImageNet model. This
          # function constructs the entire ImageNet model but shares the
          # variables across all towers.
          loss = _tower_loss(images_splits[i], depthes_splits[i],scope, reuse_variables)
          #print(images_splits[i])
          # Reuse variables for the next tower.
          reuse_variables = True
          #B) Retain the summaries from the final tower.
          summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
          # Retain the Batch Normalization updates operations only from the
          # final tower. Ideally, we should grab the updates from all towers
          # but these stats accumulate extremely fast so we can ignore the
          # other stats from the other towers without significant detriment.
          batchnorm_updates = tf.get_collection(bn.UPDATE_OPS_COLLECTION,
                                                scope)

          #C) Calculate the gradients for the batch of data on this SUN3D
          # tower.
          grads = opt.compute_gradients(loss)
          #print(loss)
          #print(scope)
          #print(grads)
          # Keep track of the gradients across all towers.
          tower_grads.append(grads)

          #print(tower_grads)
    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.

    grads = _average_gradients(tower_grads)

    # Add a summaries for the input processing and global_step.
    summaries.extend(input_summaries)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(
            tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    # Note that we maintain a "double-average" of the BatchNormalization
    # global statistics. This is more complicated then need be but we employ
    # this for backward-compatibility with our previous models.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    variables_to_average = (tf.trainable_variables() +
                            tf.moving_average_variables())
    variables_averages_op = variable_averages.apply(variables_to_average)

    # Group all updates to into a single train op.
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, variables_averages_op,
                        batchnorm_updates_op)


    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=LOG_DEVICE_PLACEMENT))
    sess.run(init)
    #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #run_metadata = tf.RunMetadata()

    
    if PRETRAINED_MODEL_CHECKPOINT_PATH:
      assert tf.gfile.Exists(PRETRAINED_MODEL_CHECKPOINT_PATH)
      tf.train.Saver.restore(sess, PRETRAINED_MODEL_CHECKPOINT_PATH)
      print('%s: Pre-trained model restored from %s' %
            (datetime.now(), PRETRAINED_MODEL_CHECKPOINT_PATH))

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

   # summary_writer = tf.summary.FileWriter(TRAIN_LOG, graph_def=sess.graph.as_graph_def(add_shapes=True))
    summary_writer = tf.summary.FileWriter(TRAIN_LOG, sess.graph)


    for step in range(NUM_ITER):
      start_time = time.time()
     #_, loss_value = sess.run([train_op, loss],options=run_options, run_metadata=run_metadata)
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        examples_per_sec = BATCH_SIZE / float(duration)
        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print(format_str % (datetime.now(), step, loss_value,
                            examples_per_sec, duration))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 5000 == 0 or (step + 1) == NUM_ITER:
        checkpoint_path = os.path.join(TRAIN_LOG, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        
def main(argv=None):
  train()

if __name__ == '__main__':
  tf.app.run()
