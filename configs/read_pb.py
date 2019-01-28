#! /usr/bin/env python3

import config_pb2
import sys
from google.protobuf import text_format

config = config_pb2.FaceLandmarkTrainingConfig()

f = open('written.2.config', "r")
text_format.Parse(f.read(), config)
f.close()



print(config.eval_record_path)
print(config.learning_rates)
print(config.train_sigma)
print(config.learning_epoches)