#!/usr/bin/env python3
import argparse
import os
import sys
import hashlib

from gpu_selector import gpu_selector
gpu_selector.auto_select(1, verbose=True)

import numpy as np
import tensorflow as tf

from face_landmark import model_light as model
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
                num_modules=1,
                learning_rate=0.0001,
                override_face_detector=None,
                train_path='/barn2/yuan/datasets/300wlp_20181002.tfrecord',
                eval_path='/barn2/yuan/datasets/aflw2000_3d_20181002.tfrecord',
                infer_graph=False,
                train_batch_size=24,
                eval_batch_size=24,
                train_sigma=5.0,
                eval_sigma=5.0,
                override_loss_op=None,
                override_train_op=None,
                checkpoint_path=None,
                save_path='saved_model/face_landmark',
                train_loss_interval=20,
                eval_interval=100,
                epoch_num_offset=0):
        self.graph = tf.Graph()
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
                                                    sigma=train_sigma, is_eval=False)
                    self.train_iterator = self.train_dataset.make_initializable_iterator()
                    self.train_image_tensor, self.train_heatmap_groundtruth_tensor = self.train_iterator.get_next()
                    self.train_heatmap_inferred_tensors = model.fan(x=self.train_image_tensor, num_modules=num_modules,
                                                                    reuse=reuse, training=True)
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
                                                    detector=self.detector.predict,is_eval=True)
                    self.val_iterator  = self.val_dataset.make_initializable_iterator()
                    self.val_image_tensor, self.val_landmark_groundtruth_tensor, self.val_bbx_tensor = self.val_iterator.get_next()
                    self.val_heatmap_inferred_tensors = model.fan(x=self.val_image_tensor, num_modules=num_modules,
                                                                    reuse=reuse, training=False)
                    assert isinstance(eval_interval, int)
                    self.eval_interval = eval_interval
                    reuse = True

                if infer_graph:
                    self.infer_image_tensor_processed = tf.placeholder(dtype=tf.uint8, shape=[None, None, None, 3], name='image_tensor')
                    image = (tf.cast(x=self.infer_image_tensor_processed, dtype=tf.float32) * 1.0 / 255.0)
                    image = tf.image.resize_images(image, size=(256, 256))
                    image = image * 2 - 1.0
                    self.infer_heatmap_inferred_tensors = model.fan(x=image, num_modules=num_modules, reuse=reuse, training=False)
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

    def epoch_train_and_eval(self, log_filename: str):
        log_filepath = os.path.join(CUR_DIR, 'cache', log_filename)
        with self.graph.as_default():
            with self.sess.as_default():
                self.sess.run(self.train_iterator.initializer)
                loss = 0.0

                step_cnt = 0

                while True:
                    step_cnt += 1
                    try:
                        loss += self.step_train()
                    except tf.errors.OutOfRangeError:
                        self.current_epoch += 1
                        self.save(os.path.join(self.save_path, str(self.current_epoch) + '-step-' + str(step_cnt)))
                        break

                    if step_cnt % self.train_loss_interval == 0:
                        print_to_file('epoch:',  self.current_epoch , 'step_cnt:', step_cnt, 'loss:', loss / self.train_loss_interval, filename=log_filepath)
                        loss = 0.0
                        self.save(os.path.join(self.save_path, str(self.current_epoch) + '-step-' + str(step_cnt)))

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
                        nme = metric.nme_batch(apts, apts_real)
                        print(type(nme))
                        plot_cdfs([nme], save_path=os.path.join(self.save_path, str(self.current_epoch) + '-step-' + str(step_cnt) + '.jpg'), xmax=0.12)
                        print_to_file('epoch:', self.current_epoch, 'step_cnt:', step_cnt, 'nme:', nme.mean(), filename=log_filepath)
        return


def parse_arguments(argv):
    """
    When exec this script, only training mode is enabled
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', type=str, default=None)
    parser.add_argument('--export_dir', dest='export_dir', type=str, default=None)
    parser.add_argument('--log_filename', dest='log_filename', type=str, default='training_log.txt')
    parser.add_argument('--plot_dir', dest='plot_dir', type=str, default=None)
    parser.add_argument('--train_loss_interval', dest='train_loss_interval', type=int, default=20)
    parser.add_argument('--eval_interval', dest='eval_interval', type=int, default=100)
    parser.add_argument('--freeze_graph', dest='freeze_graph', type=int, default=0)
    parser.add_argument('--freeze_graph_target_path', dest='freeze_graph_target_path', type=str, default=None)
    parser.add_argument('--freeze_graph_source_path', dest='freeze_graph_source_path', type=str, default=None)
    return parser.parse_args(argv)

def get_hashed_timestamp(timestamp) -> str:
    hashcode = hashlib.sha256(str(timestamp).encode('utf8'))
    return hashcode.hexdigest()


def main(argv):
    
    if argv.freeze_graph != 0:
        assert argv.freeze_graph_target_path is not None
        assert argv.freeze_graph_source_path is not None
        freeze_fan_graph(input_pb=argv.freeze_graph_source_path, output_pb=argv.freeze_graph_target_path)
        return
    if argv.export_dir is None:
        exit(0)

    # NOTE: THIS WILL SAVE MODEL TO <export_dir/> with <i.index>, <i.meta>, <i.data-00000-of-00001>

    timestamp = datetime.datetime.now().timestamp()
    hashcode = get_hashed_timestamp(timestamp)

    if argv.export_dir is None:
        export_dir = os.path.join(os.environ['SAVED_MODEL_PATH'], 'landmark' , '3dfan-' + hashcode)
    else:
        export_dir = argv.export_dir + '-' + hashcode

    if not os.path.isdir(export_dir):
        print('WARNING: the export directory does not exist, creating one:', export_dir)
        os.makedirs(name=export_dir, mode=0o777, exist_ok=False)

    fan = FAN(checkpoint_path=argv.checkpoint_path, 
        train_loss_interval=argv.train_loss_interval, 
        eval_interval=argv.eval_interval,
        learning_rate=0.0001,
        save_path=export_dir)
    for i in range(3):
        fan.epoch_train_and_eval(log_filename=argv.log_filename)

    fan = FAN(checkpoint_path=os.path.join(export_dir, str(2)), 
        train_loss_interval=argv.train_loss_interval, 
        eval_interval=argv.eval_interval,
        learning_rate=0.00005,
        epoch_num_offset=3)
    for i in range(10):
        fan.epoch_train_and_eval(log_filename=argv.log_filename)

    fan = FAN(checkpoint_path=os.path.join(argv.export_dir, str(3+10)), 
        train_loss_interval=argv.train_loss_interval, 
        eval_interval=argv.eval_interval,
        learning_rate=0.00002,
        epoch_num_offset=3+10)
    for i in range(15):
        fan.epoch_train_and_eval(log_filename=argv.log_filename)

    fan = FAN(checkpoint_path=os.path.join(argv.export_dir, str(3+10+15)), 
        train_loss_interval=argv.train_loss_interval, 
        eval_interval=argv.eval_interval,
        learning_rate=0.00001,
        epoch_num_offset=3+10+15)
    for i in range(20):
        fan.epoch_train_and_eval(log_filename=argv.log_filename)

    return


if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)
