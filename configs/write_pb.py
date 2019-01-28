#! /usr/bin/env python3

import config_pb2
import sys
from google.protobuf import text_format

# This function fills in a Person message based on user input.
def set_properties(config):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    config.name = ""  # Override in sub-classes

    config.num_modules = 4

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    config.gpu_count = 1

    config.input_size = 256
    config.use_image_gradient = True
    
    config.conv1_channels = 128
    config.conv2_channels = 168
    config.conv3_channels = 204
    config.conv4_channels = 256

    config.num_landmark_pts = 68
    config.step_limit = 0

    config.learning_rates.extend([0.001, 0.0005, 0.0002, 0.0001, 0.00005, 2e-5, 1e-5])
    config.learning_epoches.extend([  5,      5,     10,     10,      10,   20,   20])

    config.initial_checkpoint_path = ""
    config.train_record_path = '/barn2/yuan/datasets/300wlp_20181002.tfrecord'
    config.eval_record_path = '/barn2/yuan/datasets/aflw2000_3d_20181002.tfrecord'

    config.train_batch_size = 24
    config.eval_batch_size = 24

    config.train_sigma = 5.0
    config.eval_sigma = 5.0

    config.export_dir = '/barn1/yuan/saved_models/3d_fan_base'
    config.log_filename = '/barn1/yuan/saved_models/3d_fan_base/training_log.txt'

    config.train_loss_interval = 20
    config.eval_interval = 100

    return config

config = config_pb2.FaceLandmarkTrainingConfig()
config = set_properties(config)
f = open('written.config', "w+")
# text_format.MessageToString(config) is equivalent to config.__str__()
tf = text_format.MessageToString(config)
f.write(tf)
f.close()