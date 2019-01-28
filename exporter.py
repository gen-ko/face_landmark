#!/usr/bin/env python3
import argparse
import os
from wheelheaps import gpu_selector
gpu_selector.auto_select(1, verbose=True)
import tensorflow as tf

from face_landmark import model

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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', dest='config_path', type=str, required=True)
    parser.add_argument('--epoch', dest='epoch', type=int, default=0)
    parser.add_argument('--with_process', dest='with_process', type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    from face_landmark.configs import build_config
    config = build_config.parse_config(args.config_path)
    main(config, args.epoch, args.with_process)

