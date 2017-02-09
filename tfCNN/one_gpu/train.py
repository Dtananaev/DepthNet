#
# File: cnn_train.py
# Date:21.01.2017
# Author: Denis Tananaev
# 
#

 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import model
import param

TRAIN_LOG=param.TRAIN_LOG
BATCH_SIZE=param.BATCH_SIZE
NUM_ITER=param.NUM_ITER
LOG_DEVICE_PLACEMENT=param.LOG_DEVICE_PLACEMENT
def train():
  """Train SUN3D for a number of steps."""
  with tf.Graph().as_default(),tf.device('/gpu:1'):
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for SUN3D.
    images, depths = model.inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    phase_train=True
    
    scores = model.inference(images,phase_train)
    # Calculate loss.
    loss = model.loss(scores, depths)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model.train(loss, global_step)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96)
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        self._start_time = time.time()
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        duration = time.time() - self._start_time
        loss_value = run_values.results
        if self._step % 10 == 0:
          num_examples_per_step = BATCH_SIZE
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(checkpoint_dir=TRAIN_LOG,
hooks=[tf.train.StopAtStepHook(last_step=NUM_ITER),tf.train.NanTensorHook(loss),_LoggerHook()],config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options,log_device_placement=LOG_DEVICE_PLACEMENT)) as mon_sess:
        while not mon_sess.should_stop():
            print(mon_sess.run(loss))
            mon_sess.run(train_op )

            
def main(argv=None):
  train()


if __name__ == '__main__':
  tf.app.run() 
 
