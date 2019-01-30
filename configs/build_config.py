#! /usr/bin/env python3

from face_landmark.configs import config_pb2
import sys
import os
from google.protobuf import text_format

def parse_config(config_path):
    if not os.path.exists(config_path):
        raise ValueError(f'{config_path} does not exist.')
    config = config_pb2.FaceLandmarkTrainingConfig()
    f = open(config_path, "r")
    text_format.Parse(f.read(), config)
    f.close()

    if config.num_modules == 0:
        config.num_modules = 4

    if config.gpu_count == 0:
        config.gpu_count = 1

    if config.input_size == 0:
        config.input_size = 256

    if config.conv1_channels == 0:
        config.conv1_channels = 128

    if config.conv2_channels == 0:
        config.conv2_channels = 168

    if config.conv3_channels == 0:
        config.conv3_channels = 204

    if config.conv4_channels == 0:
        config.conv4_channels = 256

    if config.num_landmark_pts == 0:
        config.num_landmark_pts = 68


    assert len(config.learning_rates) == len(config.learning_epoches), f'learning rates and epoches must have equal length'
    if len(config.learning_rates) == 0:
        config.learning_rates.extend([0.001, 0.0005, 0.0002, 0.0001, 0.00005, 2e-5, 1e-5])
        config.learning_epoches.extend([  5,      5,     10,     10,      10,   20,   20])

    if config.initial_checkpoint_path == "":
        config.initial_checkpoint_path = ""

    if config.train_record_path == "":
        config.train_record_path = "/barn2/yuan/datasets/300wlp_20181002.tfrecord"

    if config.eval_record_path == "":
        config.eval_record_path = "/barn2/yuan/datasets/aflw2000_3d_20181002.tfrecord"

    if config.train_batch_size == 0:
        config.train_batch_size = 24

    if config.eval_batch_size == 0:
        config.eval_batch_size = 24

    if config.train_sigma == 0:
        config.train_sigma = 5.0
        
    if config.eval_sigma == 0:
        config.eval_sigma = 5.0

    if config.export_dir == "":
        raise ValueError('export dir must be explicitly specified in config file')

    if config.log_filename == "":
        config.log_filename = os.path.join(config.export_dir, 'training_log.txt')

    if os.path.exists(config.log_filename):
        pass
        #raise ValueError('logging filepath already exists')


    if config.plot_dir == "":
        config.plot_dir = os.path.join(config.export_dir, 'figures')
    if not os.path.exists(config.plot_dir):
        pass
        #os.makedirs(name=config.plot_dir, mode=0o777, exist_ok=False)

    if config.train_loss_interval == 0:
        config.train_loss_interval = 20

    if config.eval_interval == 0:
        config.eval_interval = 100

    return config

if __name__ == '__main__':
    config = parse_config('default_config.txt')
    print(config.__str__())