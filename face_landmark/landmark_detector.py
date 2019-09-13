#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import cv2
from lambdaflow.python.face_landmark import preprocess_tf as landmark_util
# NOTE: the next generation of preprocess should be imported from lambdaflow.python.utils
import time

CUR_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

class DetectionError(Exception):
    pass

class LandmarkDetector(object):
    def __init__(self, model_path=None):
        if model_path is None:
            raise ValueError('model_path is None')
        model_path = os.path.join(model_path)
        #gpu_options = 
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        #config.gpu_options.allow_growth = True
        graph = tf.Graph()
        with graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')
            self.sess = tf.Session(config=config, graph=graph)
            with self.sess.as_default():
                self.x = graph.get_tensor_by_name('fan_input_tensor:0')
                self.y = graph.get_tensor_by_name('fan_output_tensor:0')[0]
                self.image_size = self.x.shape[1].value
                print(f'fan input shape: {self.x.shape}')
                print(f'fan output shape: {self.y.shape}')
                self.image = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
                self.box = tf.placeholder(dtype=tf.float32, shape=4)

                self.heatmap = tf.placeholder(dtype=tf.float32, shape=[self.y.shape[0].value, self.y.shape[1].value, self.y.shape[2].value, self.y.shape[3].value])

                image, trans_matrix = landmark_util.infer_preprocess(image=self.image, bbx=self.box)
                self.preprocessed_image = image
                self.trans_matrix = trans_matrix
                self.pts = landmark_util.heatmap2pts_batch(self.heatmap, target_height=self.image_size, target_width=self.image_size)[0]
                self.landmark = landmark_util.infer_postprocess(landmark=self.pts, trans_matrix=trans_matrix)
        return

    def predict(self, x):
        start = time.time()
        ret = self.sess.run(self.y, {self.x: x})
        elapsed = time.time() - start
        print(f'elapsed time: {int(elapsed*1000)} ms')
        return ret

    def preprocess(self, raw_image, face_detector):
        box = face_detector.predict_1x1(raw_image)
        image, trans_matrix = self.sess.run([self.preprocessed_image, self.trans_matrix], {self.image: raw_image, self.box: box})
        return image, trans_matrix

    def postprocess(self, y, trans_matrix):
        landmark = self.sess.run(self.landmark, {self.heatmap:y, self.trans_matrix: trans_matrix})
        return landmark

    def infer(self, raw_image, face_detector):
        h, w, _ = raw_image.shape
        image, trans_matrix = self.preprocess(raw_image, face_detector)
        image = image.astype(np.float32) / (255.0 / 2.0) - 1.0 
        image = cv2.resize(image, (self.image_size, self.image_size))
        y = self.predict([image])

        landmark = self.postprocess(y, trans_matrix)
        landmark = (landmark * np.array([w, h])).astype(int)
        return landmark



if __name__ == '__main__':
    exit(0)
