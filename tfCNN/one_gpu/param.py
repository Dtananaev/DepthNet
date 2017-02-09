#
# File: param.py
# Date:21.01.2017
# Author: Denis Tananaev
# 
#
#the dataset file for training set


tfrecords_filename_train ='/misc/lmbraid11/tananaed/DepthNet/tfCNN/data_processing/dataset/train_sun3d.tfrecords'
tfrecords_filename_test ='/misc/lmbraid11/tananaed/DepthNet/tfCNN/data_processing/dataset/test_sun3d.tfrecords'
#the parameters of dataset
IMAGE_SIZE_W=256
IMAGE_SIZE_H=192
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=6553
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1253
INVERSE_DEPTH=False #inverse values of the depth images
FLOAT16=False
#the parameters data upload
BATCH_SIZE=32
NUM_PREPROCESS_THREADS=4
NUM_READERS=1
INPUT_QUEUE_MEMORY_FACTOR=1
DATA_SHUFFLE=True
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.995     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 30.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0001       # Initial learning rate.
WEIGHT_DECAY=0.000001
#training and test
NUM_ITER=3000000
TRAIN_LOG="./log"
TEST_LOG="./eval"
LOG_DEVICE_PLACEMENT=False
PRETRAINED_MODEL_CHECKPOINT_PATH=''
EVAL_RUN_ONCE=True
EVAL_INTERVAL_SECS=5*60 #5 minutes


