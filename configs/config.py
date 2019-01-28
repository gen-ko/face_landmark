"""
3D FAN
Base Configurations class.

Copyright (c) 2019 Blacksesame Technologies, Inc.
Licensed under the NO License
Written by Yuan Liu
"""

import numpy as np


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes

    NUM_MODULES = 4

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1

    INPUT_SIZE = 256

    USE_IMAGE_GRADIENT = True

    CONV1_CHANNELS = 128
    CONV2_CHANNELS = 168
    CONV3_CHANNELS = 204
    CONV4_CHANNELS = 256

    NUM_LANDMARK_PTS = 68

    STEP_LIMIT = 0

    LEARNING_RATES = [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 2e-5, 1e-5]
    LEARNING_EPOCHES = [  5,      5,     10,     10,      10,   20,   20]

    INITIAL_CHECKPOINT_PATH = None

    TRAIN_PATH = '/barn2/yuan/datasets/300wlp_20181002.tfrecord'
    EVAL_PATH = '/barn2/yuan/datasets/aflw2000_3d_20181002.tfrecord'

    TRAIN_BATCH_SIZE = 24
    EVAL_BATCH_SIZE = 24

    TRAIN_SIGMA = 5.0
    EVAL_SIGMA = 5.0


    EXPORT_DIR = '/barn1/yuan/saved_models/3d_fan_base'
    LOG_FILENAME = '/barn1/yuan/saved_models/3d_fan_base/training_log.txt'

    # steps between printing the training losses
    TRAIN_LOSS_INTERVAL=20
    EVAL_INTERVAL=100

    PLOT_DIR=None
    




