#!/usr/bin/env python3
import argparse
import os
import sys
import hashlib

from gpu_selector import gpu_selector
gpu_selector.auto_select(1, verbose=True)

import numpy as np
import tensorflow as tf

from face_landmark import model_with_config as model
from face_detection import face_detector
from face_landmark import inputs
from face_landmark import preprocess
from face_landmark import preprocess_tf
from face_landmark import metric
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import time
import datetime  # timestamp
import pdb


CUR_DIR = os.path.dirname(os.path.realpath(__file__))
'''
def plot_base(v1, label, save_path):
    #plt.figure(1)
    label_1 = 'train' + ' ' + label
    line_1, = plt.plot(v1, label=label_1)
    plt.legend(handles=[line_1])
    plt.xlabel('epoch')
    plt.ylabel(label)
    #plt.title(self.titletext)
    plt.savefig(save_path)
    plt.close()
    return

def plot_metric(metric_train: list, filename: str=None):
    if filename is None:
        filename = 'metric_plot.png'
    v1 = metric_train
    plot_base(v1, 'loss', filename)
    return
'''


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def plot_cdfs(xs, save_path=None, xmax=None):
    plt.clf()
    for x in xs:
        num_bins = x.shape[0]
        counts, bin_edges = np.histogram (x, bins=num_bins)
        cdf = np.cumsum (counts)

        x_axis = bin_edges[1:]
        x_axis = np.insert(x_axis, 0, x_axis[0])
        x_axis = np.insert(x_axis, 0, 0)

        y_axis = cdf/cdf[-1]
        y_axis = np.insert(y_axis, 0, 0)
        y_axis = np.insert(y_axis, 0, 0)

        n_draw = int(min(find_nearest_idx(y_axis, 0.9) * 2, y_axis.shape[0]*0.98))
        x_axis = x_axis[:n_draw] * 100
        y_axis = y_axis[:n_draw] * 100

        if xmax is None:
            xmax = np.max(x_axis) / 100.0
        plt.xlim(0, xmax * 100)
        plt.ylim(0, 100)
        plt.ylabel('sample %')
        plt.xlabel('nme %')

        #x_axis_finegrain = np.linspace(x_axis.min(), x_axis.max(), int(90000 / num_bins) + num_bins)
        #y_axis_smooth = BSpline(x_axis ,y_axis,x_axis_finegrain)
        plt.plot(x_axis, y_axis)
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path)
    return


def print_to_file(*args, **kwargs):
    """a print() wrapper that prints to stdout along with a log file at the same time"""
    filename = kwargs['filename']
    # also print to stdout
    print(*args)
    with open(filename, 'a', encoding='utf-8') as file:
        print(*args, file=file)
    return

class Detector(object):
    """A wrapper class for object detection in landmark use"""
    def __init__(self):
        self.d = face_detector.FaceDetector()

    def predict(self, x):
        try:
            bbx = self.d.predict_1x1_raise(x, threshold=0.87)
        except face_detector.DetectionError:
            bbx = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        return bbx

def freeze_fan_graph(input_pb, output_pb):
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
                x = tf.placeholder(dtype=tf.uint8, shape=[None, None, None, 3], name='image_tensor_raw')
                bbx = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='bbx_raw')

                image = x[0]
                bbx0 = bbx[0]
                image, trans_matrix = preprocess_tf.infer_preprocess(image, bbx0)
                shape = tf.shape(image)
                h = shape[0]
                w = shape[1]
                c = shape[2]

                image = tf.reshape(image, [1, h, w, 3])

                # resize image
                tmp = (tf.cast(image, dtype=tf.float32) / 255.0)
                tmp = tf.image.resize_images(tmp, size=[256, 256], method=tf.image.ResizeMethod.BILINEAR)
                tmp = tmp * 2.0 - 1
                print(tmp.shape)
                tmp = model.fan(x=tmp, num_modules=1, reuse=False, training=False)[-1]

                # convert heatmap to landmark coordinates
                tmp = preprocess_tf.heatmap2pts(tmp)
                tmp = tmp[0]
                tmp = preprocess_tf.infer_postprocess(tmp, trans_matrix)
                tmp = tf.reshape(tmp, [-1, 68, 2])

                pts_tensor = tf.identity(tmp, name='pts_tensor')

                saver = tf.train.Saver()
                saver.restore(sess, input_pb)

                frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                    sess=sess,
                    input_graph_def=graph.as_graph_def(),
                    output_node_names=['pts_tensor'])

                export_dir, export_pb_name = os.path.split(output_pb)

                if not os.path.isdir(export_dir):
                    print('WARNING: the export directory does not exist, creating one:', export_dir)
                    os.makedirs(name=export_dir, mode=0o777, exist_ok=False)

                if os.path.isfile(output_pb):
                    print('WARNING: the output filepath exists, overwriting it...', output_pb)
                    
                tf.train.write_graph(frozen_graph_def , export_dir, export_pb_name, as_text=False)
    return

class FAN(object):
    """Refer to the original paper https://arxiv.org/pdf/1703.07332.pdf"""
    def __init__(self,
                config,
                learning_rate=0.0001,
                override_face_detector=None,
                infer_graph=False,
                override_loss_op=None,
                override_train_op=None,
                checkpoint_path=None,
                save_path='saved_model/face_landmark',
                epoch_num_offset=0):
        self.graph = tf.Graph()
        num_modules = config.NUM_MODULES
        train_path = config.TRAIN_PATH
        eval_path = config.EVAL_PATH
        train_batch_size = config.TRAIN_BATCH_SIZE
        eval_batch_size = config.EVAL_BATCH_SIZE

        train_sigma = config.TRAIN_SIGMA
        eval_sigma = config.EVAL_SIGMA

        train_loss_interval = config.TRAIN_LOSS_INTERVAL
        eval_interval = config.EVAL_INTERVAL

        with self.graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                if override_face_detector is None:
                    self.detector = Detector()
                else:
                    self.detector = override_face_detector
                reuse = False
                if train_path is not None:
                    self.train_dataset = inputs.input_fn(train_path,
                                                    batch_size=train_batch_size,
                                                    detector=self.detector.predict,
                                                    sigma=train_sigma, is_eval=False,
                                                    input_size=config.INPUT_SIZE,
                                                    num_landmark_pts=config.NUM_LANDMARK_PTS)
                    self.train_iterator = self.train_dataset.make_initializable_iterator()
                    self.train_image_tensor, self.train_heatmap_groundtruth_tensor = self.train_iterator.get_next()
                    self.train_heatmap_inferred_tensors = model.fan(config=config, x=self.train_image_tensor, reuse=reuse, training=True)
                    if override_loss_op is None:
                        self.loss_op = 0.0
                        for tensor in self.train_heatmap_inferred_tensors:
                            self.loss_op = self.loss_op + tf.losses.mean_squared_error(labels=tensor, predictions=self.train_heatmap_groundtruth_tensor)
                    else:
                        self.loss_op = override_loss_op(self.train_heatmap_inferred_tensors, self.train_heatmap_groundtruth_tensor)
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        if override_train_op is None:
                            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_op)
                        else:
                            self.train_op = override_train_op(self.loss_op)
                    assert isinstance(train_loss_interval, int)
                    self.train_loss_interval = train_loss_interval
                    reuse = True

                if eval_path is not None:
                    self.val_dataset = inputs.input_fn(eval_path,
                                                    batch_size=eval_batch_size,
                                                    detector=self.detector.predict,is_eval=True,
                                                    input_size=config.INPUT_SIZE,
                                                    num_landmark_pts=config.NUM_LANDMARK_PTS)
                    self.val_iterator  = self.val_dataset.make_initializable_iterator()
                    self.val_image_tensor, self.val_landmark_groundtruth_tensor, self.val_bbx_tensor = self.val_iterator.get_next()
                    self.val_heatmap_inferred_tensors = model.fan(config=config, x=self.val_image_tensor,
                                                                    reuse=reuse, training=False)
                    assert isinstance(eval_interval, int)
                    self.eval_interval = eval_interval
                    reuse = True

                if infer_graph:
                    self.infer_image_tensor_processed = tf.placeholder(dtype=tf.uint8, shape=[None, None, None, 3], name='image_tensor')
                    image = (tf.cast(x=self.infer_image_tensor_processed, dtype=tf.float32) * 1.0 / 255.0)
                    image = tf.image.resize_images(image, size=(256, 256))
                    image = image * 2 - 1.0
                    self.infer_heatmap_inferred_tensors = model.fan(config=config, x=image, reuse=reuse, training=False)
                    reuse = True

                self.saver = tf.train.Saver()
                if checkpoint_path is not None:
                    init_op = tf.global_variables_initializer()
                    self.sess.run(init_op)
                    self.saver.restore(self.sess, checkpoint_path)
                else:
                    init_op = tf.global_variables_initializer()
                    self.sess.run(init_op)
                self.save_path = save_path
        self.current_epoch = epoch_num_offset
        return

    def load(self, path):
        """Load the checkpoint of current graph.
        path is the target filename prefix,
        for example,
        path = 'foo/bar'
        then the checkpoint will be loaded from
        'foo/bar.idex, foo/bar.meta, foo/bar.data-00000-of-00001'
        If the checkpoint has variables with the same name but different op type,
        it is possible that the load will fail.
        """
        with self.graph.as_default():
            with self.sess.as_default():
                self.saver.restore(self.sess, path)
        return

    def save(self, path):
        """Save the checkpoint of current graph.
        path is the destination filename prefix,
        for example,
        path = 'foo/bar'
        then the checkpoint will be saved as
        'foo/bar.index, foo/bar.meta, foo/bar.data-00000-of-00001'
        """
        with self.graph.as_default():
            with self.sess.as_default():
                self.saver.save(self.sess, path)
        return

    def step_train(self):
        with self.graph.as_default():
            with self.sess.as_default():
                loss, _ = self.sess.run([self.loss_op, self.train_op])
        return loss

    def step_eval(self):
        with self.graph.as_default():
            with self.sess.as_default():
                heatmaps, pts_real, bbx = self.sess.run([self.val_heatmap_inferred_tensors[-1],
                                                         self.val_landmark_groundtruth_tensor,
                                                         self.val_bbx_tensor])
                pts = preprocess.heatmap2pts(heatmaps)
                #d = np.sqrt((bbx[:,2] - bbx[:,0]) * (bbx[:,3] - bbx[:,1]))
                d = np.repeat([1.0], pts.shape[0])
        return pts.tolist(), pts_real.tolist(), d.tolist()

    def infer(self, image):
        with self.graph.as_default():
            with self.sess.as_default():
                bbx = self.detector.predict(image)
                image_processed, trans_matrix = preprocess.infer_preprocess(image, bbx)
                heatmaps = self.sess.run(self.infer_heatmap_inferred_tensors[-1], {self.infer_image_tensor_processed: [image_processed]})[0]
                pts = preprocess.heatmap2pts(heatmaps)
                pts_raw = preprocess.infer_postprocess(pts, trans_matrix)
        return pts_raw, bbx

    def epoch_train_and_eval(self, log_filename: str, step_limit=None):
        log_filepath = log_filename
        with self.graph.as_default():
            with self.sess.as_default():
                self.sess.run(self.train_iterator.initializer)
                loss = 0.0

                step_cnt = 0

                while True and (step_cnt < step_limit or step_limit == 0):
                    step_cnt += 1
                    try:
                        loss += self.step_train()
                    except tf.errors.OutOfRangeError:
                        break

                    if step_cnt % self.train_loss_interval == 0:
                        print_to_file('epoch:',  self.current_epoch , 'step_cnt:', step_cnt, 'loss:', loss / self.train_loss_interval, filename=log_filepath)
                        loss = 0.0
                        self.save(os.path.join(self.save_path, str(self.current_epoch) + '-step-' + str(step_cnt) + '.cp'))

                    if step_cnt % self.eval_interval == 0:
                        self.sess.run(self.val_iterator.initializer)
                        apts = []
                        apts_real = []
                        ad = []
                        while True:
                            try:
                                pts, pts_real, d = self.step_eval()
                                apts.extend(pts)
                                apts_real.extend(pts_real)
                                ad.extend(d)
                            except tf.errors.OutOfRangeError:
                                break
                        apts = np.array(apts)
                        apts_real = np.array(apts_real)
                        ad = np.array(ad)
                        # NOTE: API to be modified to consistent with non-batch version
                        nme = metric.nme_batch(apts_real, apts)
                        print(type(nme))
                        plot_cdfs([nme], save_path=os.path.join(self.save_path, str(self.current_epoch) + '-step-' + str(step_cnt) + '.jpg'), xmax=0.12)
                        print_to_file('epoch:', self.current_epoch, 'step_cnt:', step_cnt, 'nme:', nme.mean(), filename=log_filepath)
                self.current_epoch += 1
                self.save(os.path.join(self.save_path, str(self.current_epoch) + '.cp'))
        return


def main(config):
    if not os.path.isdir(config.EXPORT_DIR):
        print('WARNING: the export directory does not exist, creating one:', config.EXPORT_DIR)
        os.makedirs(name=config.EXPORT_DIR, mode=0o777, exist_ok=False)

    cur_epoch = 0
    for epoch, lr in zip(config.LEARNING_EPOCHES, config.LEARNING_RATES):
        if cur_epoch == 0:
            checkpoint_path = config.INITIAL_CHECKPOINT_PATH
        else:
            checkpoint_path = os.path.join(config.EXPORT_DIR, str(cur_epoch)+'.cp')

        fan = FAN(config=config,
                  checkpoint_path=checkpoint_path, 
                  learning_rate=lr,
                  save_path=config.EXPORT_DIR,
                  epoch_num_offset=cur_epoch)
        for i in range(epoch):
            fan.epoch_train_and_eval(log_filename=config.LOG_FILENAME, step_limit=config.STEP_LIMIT)
            cur_epoch += 1
    return


if __name__ == '__main__':
    from face_landmark.configs.config_light import ConfigLight
    config = ConfigLight()
    main(config)
