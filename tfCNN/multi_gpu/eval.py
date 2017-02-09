

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, '../layers/')


from datetime import datetime
import math
import os.path
import time
import numpy as np
import tensorflow as tf
import param
import model
import data
import evalfunct as evaluate
CHECKPOINT_DIR=param.TRAIN_LOG
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL=param.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
BATCH_SIZE=param.BATCH_SIZE
MOVING_AVERAGE_DECAY=param.MOVING_AVERAGE_DECAY
TEST_LOG=param.TEST_LOG
EVAL_RUN_ONCE=param.EVAL_RUN_ONCE
EVAL_INTERVAL_SECS=param.EVAL_INTERVAL_SECS

def _eval_once(saver, summary_writer, scale_inv_error, L1_relative_error,L1_inverse_error, summary_op):
  """Runs Eval once."""
  
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
      if os.path.isabs(ckpt.model_checkpoint_path):
        # Restores from checkpoint with absolute path.
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        # Restores from checkpoint with relative path.
        saver.restore(sess, os.path.join(CHECKPOINT_DIR,
                                         ckpt.model_checkpoint_path))

      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/log/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('Succesfully loaded model from %s at step=%s.' %
            (ckpt.model_checkpoint_path, global_step))
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / BATCH_SIZE))
      # Counts the number of correct predictions.
      count_scale_inv_error = 0.0
      count_L1_relative_error = 0.0
      count_L1_inverse_error = 0.0
      total_sample_count = num_iter * BATCH_SIZE
      step = 0

      print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
      start_time = time.time()
      while step < num_iter and not coord.should_stop():
        scale_inv_error, L1_relative_error,L1_inverse_error = sess.run([scale_inv_error, L1_relative_error,L1_inverse_error])
        
        count_scale_inv_error += scale_inv_error
        count_L1_relative_error += L1_relative_error
        count_L1_inverse_error += L1_inverse_error        
        
        step += 1
        if step % 20 == 0:
          duration = time.time() - start_time
          sec_per_batch = duration / 20.0
          examples_per_sec = BATCH_SIZE / sec_per_batch
          print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                'sec/batch)' % (datetime.now(), step, num_iter,
                                examples_per_sec, sec_per_batch))
          start_time = time.time()

      # Compute precision @ 1.
      result_scale_inv_error = count_scale_inv_error / total_sample_count
      result_L1_relative_error = count_L1_relative_error / total_sample_count
      result_L1_inverse_error = count_L1_inverse_error / total_sample_count      
      
      print('%s: sc-inv = %.4f L1-rel = %.4f L1-inv = %.4f  [%d examples]' %
            (datetime.now(), result_scale_inv_error, result_L1_relative_error,result_L1_inverse_error, total_sample_count))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='sc-inv ', simple_value=result_scale_inv_error)
      summary.value.add(tag='L1-rel ', simple_value=result_L1_relative_error)
      summary.value.add(tag='L1-inv ', simple_value=result_L1_inverse_error)      
      summary_writer.add_summary(summary, global_step)

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
    
def evaluate():
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default():
    #1 Get images and depths from the dataset.
    images, Depths = data.read_sun3d(eval_data=True)

    #2 Build a Graph that computes the logits predictions from the
    # inference model.
    result = model.inference(images, phase_train=False)
    
    # Calculate errors.
    scale_inv_error=evaluate.scinv(result,Depths)
    L1_relative_error=evaluate.L1rel(result,Depths)
    L1_inverse_error=evaluate.L1inv(result,Depths)
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.train.SummaryWriter(TEST_LOG,
                                            graph_def=graph_def)

    while True:
      _eval_once(saver, summary_writer, scale_inv_error, L1_relative_error,L1_inverse_error, summary_op)
      if EVAL_RUN_ONCE:
        break
      time.sleep(EVAL_INTERVAL_SECS)
      
      
def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run() 