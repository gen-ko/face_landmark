#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

from face_landmark import model
from face_detection import face_detector
from face_landmark import inputs
from face_landmark import preprocess
from face_landmark import preprocess_tf
from face_landmark import metric
import time

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

class FAN(object):
    """Refer to the original paper https://arxiv.org/pdf/1703.07332.pdf"""
    def __init__(self, 
                num_modules=4,
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
                save_path='saved_model/face_landmark'):
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

                    reuse = True

                if eval_path is not None:
                    self.val_dataset = inputs.input_fn(eval_path,
                                                    batch_size=eval_batch_size,
                                                    detector=self.detector.predict,is_eval=True)
                    self.val_iterator  = self.val_dataset.make_initializable_iterator()
                    self.val_image_tensor, self.val_landmark_groundtruth_tensor, self.val_bbx_tensor = self.val_iterator.get_next()
                    self.val_heatmap_inferred_tensors = model.fan(x=self.val_image_tensor, num_modules=num_modules,
                                                                    reuse=reuse, training=False)
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
                    self.saver.restore(self.sess, checkpoint_path)
                else:
                    init_op = tf.global_variables_initializer()
                    self.sess.run(init_op)
                self.save_path = save_path
        self.current_epoch = 0
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


    def freeze(self, path):
        with self.graph.as_default():
            with self.sess.as_default():
                tf.identity(self.infer_heatmap_inferred_tensors[-1], name='y')


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
                tmp = model.fan(x=tmp, num_modules=4, reuse=True, training=False)[-1]

                # convert heatmap to landmark coordinates
                tmp = preprocess_tf.heatmap2pts(tmp)
                tmp = tmp[0]
                tmp = preprocess_tf.infer_postprocess(tmp, trans_matrix)
                tmp = tf.reshape(tmp, [-1, 68, 2])

                pts_tensor = tf.identity(tmp, name='pts_tensor')

                frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                    sess=self.sess,
                    input_graph_def=self.graph.as_graph_def(),
                    output_node_names=['pts_tensor'])



                export_dir = path

                if os.path.isdir(export_dir):
                    import shutil
                    shutil.rmtree(export_dir)

                tf.train.write_graph(frozen_graph_def , export_dir, 'frozen_graph.pb', as_text=False)
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
                n_steps = 0
                step_cnt = 0

                while True:
                    step_cnt += 1
                    try:
                        loss += self.step_train()
                        n_steps += 1
                    except tf.errors.OutOfRangeError:
                        n_steps += 1
                        self.current_epoch += 1
                        self.save(self.save_path + str(self.current_epoch))
                        break

                    if step_cnt % 20 == 0:
                        print_to_file('epoch:',  self.current_epoch , 'step_cnt:', step_cnt, 'loss:', loss / n_steps, filename=log_filepath)
                        loss = 0.0
                        n_steps = 0

                    if step_cnt % 100 == 0:
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
                        nme = metric.nme_batch(apts, apts_real, ad)
                        print_to_file('epoch:', self.current_epoch, 'step_cnt:', step_cnt, 'nme:', nme, filename=log_filepath)
        return


def parse_arguments(argv):
    """
    When exec this script, only training mode is enabled
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', type=str, default=None)
    parser.add_argument('--export_dir', dest='export_dir', type=str, default=None)
    parser.add_argument('--log_filename', dest='log_filename', type=str, default='training_log.txt')
    return parser.parse_args()

def main(argv):
    if argv.export_dir is None:
        exit(0)
    fan = FAN(checkpoint_path=argv.checkpoint_path)
    for i in range(100):
        fan.epoch_train_and_eval(log_filename=argv.log_filename)
        fan.save(os.path.join(argv.export_dir, str(i)))
    return


if __name__ == '__main__':
    argv = parse_arguments(sys.argv)
    main(argv)
