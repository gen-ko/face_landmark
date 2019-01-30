#!/usr/bin/env python3
import argparse
import os
from utils import gpu_selector
gpu_selector.auto_select(1, verbose=True)
import tensorflow as tf
import numpy as np
from face_landmark import model
from face_landmark import preprocess_tf as landmark_util

def main(config, epoch, with_process):
    if epoch == 0:
        total_epoches = 0
        for epoch_i in config.learning_epoches:
            total_epoches += epoch_i
    else:
        total_epoches = epoch
    with tf.Session() as sess:
        if not with_process:
            x = tf.placeholder(dtype=tf.float32, shape=[1, config.input_size, config.input_size, 3], name='fan_input_tensor')
        else:
            raise NotImplementedError
        fan = model.fan(config=config, x=x, reuse=None, training=True)
        y = tf.identity(fan, 'fan_output_tensor')

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        tvars = tf.trainable_variables()
        tvars_vals = sess.run(tvars)

        for var in tvars:
            print(var.name)

        saver = tf.train.Saver()
        path = os.path.join(config.export_dir, str(total_epoches) + '.cp')
        print(f'load from {path}')
        saver.restore(sess, path)
        print('loaded.')


        run_meta = tf.RunMetadata()


        opts = tf.profiler.ProfileOptionBuilder.float_operation()    
        flops = tf.profiler.profile(tf.get_default_graph(), run_meta=run_meta, cmd='op', options=opts)
        print(f'FLOPS: {flops.total_float_ops}')

        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=tf.get_default_graph().as_graph_def(),
            output_node_names=['fan_output_tensor'])

    export_pb_name = 'frozen_graph_epoch_' + str(epoch) + '.pb'
    tf.train.write_graph(frozen_graph_def , config.export_dir, export_pb_name, as_text=False)
    return


def export_preprocess_graph(shape, image_dtype=tf.uint8, config=None):
    """
    Export a graph for image preprocessing
    The shape is fixed
    Args:
        shape: array-like, typically [720, 1280, 3]
    Note:
        Input Tensors:
            ld_preprocess_input_image_tensor: [H, W, C]@uint8 image
            ld_preprocess_input_box_tensor: [4]@float32, normalized coordinates, (y1, x1, y2, x2)
        Output Tensors:
            ld_preprocess_output_image_tensor: [fan_input, fan_input, fan_channel]@float32
            ld_preprocess_output_trans_tensor: [3, 3]@float32
    """
    if config is None:
        raise ValueError
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        if image_dtype == tf.uint8:
            pass
        image = tf.placeholder(dtype=tf.uint8, shape=shape, name='ld_preprocess_input_image_tensor')
        box = tf.placeholder(dtype=tf.float32, shape=4, name='ld_preprocess_input_box_tensor')
        image_, trans_matrix = landmark_util.infer_preprocess(image=image, bbx=box)
        image_ = tf.cast(image_, tf.float32) / (255.0 / 2.0) - 1.0
        image_ = tf.image.resize_images(image_, size=(config.input_size, config.input_size))
        out_0 = tf.identity(image_, name='ld_preprocess_output_image_tensor')
        out_1 = tf.identity(trans_matrix, name='ld_preprocess_output_trans_tensor')

        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()    
        flops = tf.profiler.profile(tf.get_default_graph(), run_meta=run_meta, cmd='op', options=opts)
        print(f'FLOPS: {flops.total_float_ops}')

        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=tf.get_default_graph().as_graph_def(),
            output_node_names=['ld_preprocess_output_image_tensor',
                               'ld_preprocess_output_trans_tensor'])

    export_pb_name = 'frozen_graph_preprocess.pb'
    tf.train.write_graph(frozen_graph_def , config.export_dir, export_pb_name, as_text=False)
    return


def export_postprocess_graph(shape, image_dtype=tf.uint8, config=None):
    """
    Export a graph for image preprocessing
    The shape is fixed
    Args:
        shape: array-like, typically [720, 1280, 3]
    Note:
        Input Tensors:
            ld_postprocess_input_heatmap_tensor: [1, H, W, C]@float32 heatmap
            ld_postprocess_input_trans_tensor: [3,3 ]@float32
        Output Tensors:
            ld_postprocess_output_landmark_tensor: [N_Landmark_points, 2]@int64
    """
    if config is None:
        raise ValueError
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        if image_dtype == tf.uint8:
            pass
        heatmap = tf.placeholder(dtype=tf.float32, 
                                 shape=(1, config.input_size, config.input_size, config.num_landmark_pts),
                                 name='ld_postprocess_input_heatmap_tensor')
        trans_matrix = tf.placeholder(dtype=tf.float32,
                                      shape=(3,3),
                                      name='ld_postprocess_input_trans_tensor')
        pts = landmark_util.heatmap2pts_batch(heatmap, 
                                              target_height=config.input_size,
                                              target_width=config.input_size)[0]
        landmark = landmark_util.infer_postprocess(landmark=pts, trans_matrix=trans_matrix)
        landmark = landmark * np.array([shape[1], shape[0]])
        landmark = tf.cast(landmark, tf.int64)
        out_0 = tf.identity(landmark, name='ld_postprocess_output_landmark_tensor')

        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()    
        flops = tf.profiler.profile(tf.get_default_graph(), run_meta=run_meta, cmd='op', options=opts)
        print(f'FLOPS: {flops.total_float_ops}')

        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=tf.get_default_graph().as_graph_def(),
            output_node_names=['ld_postprocess_output_landmark_tensor'])

    export_pb_name = 'frozen_graph_postprocess.pb'
    tf.train.write_graph(frozen_graph_def , config.export_dir, export_pb_name, as_text=False)
    return



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', dest='export_dir', type=str, default=None)
    parser.add_argument('--config_path', dest='config_path', type=str, required=True)
    parser.add_argument('--epoch', dest='epoch', type=int, default=0)
    parser.add_argument('--with_process', dest='with_process', type=int, default=0)
    parser.add_argument('--mode', dest='mode', type=str, default='model', choices=['model', 'preprocess', 'postprocess'])
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    from face_landmark.configs import build_config
    config = build_config.parse_config(args.config_path)
    if args.export_dir is not None:
        config.export_dir = args.export_dir
    if args.mode == 'model':
        main(config, args.epoch, args.with_process)
    elif args.mode == 'preprocess':
        export_preprocess_graph(shape=(800, 800,3), config=config)
    elif args.mode == 'postprocess':
        export_postprocess_graph(shape=(800, 800, 3), config=config)

