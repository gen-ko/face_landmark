import argparse

import numpy as np
import tensorflow as tf

from face_landmark import model
from face_landmark import load_dataset

class ModelTrain(object):
    def __init__(self, num_modules=1, learning_rate=0.00001):
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(target='', graph=self.graph, config=config)
        self.image_tensor = tf.placeholder(dtype=tf.uint8, shape=[None, None, None, 3], name='image_tensor')
        x1 = (tf.cast(x=image_tensor, dtype=tf.float32) * 2.0 / 255.0) - 1.0
        x2 = tf.image.resize_images(x1, size=(256, 256))
        heatmap_inferred_tensors = model.fan(x=x2, num_modules=num_modules)
        self.num_modules = num_modules
        for i, heatmap_tensor in heatmap_inferred_tensors:
            self.heatmap_tensors.append(tf.identity(heatmap_tensor, name='heatmap_t'+str(i)+'_tensor'))
           
        heatmap_groundtruth_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 68], name='heatmap_groundtruth_tensor')
        self.target = self.heatmap_ground_truth_tensor
        self.y = self.heatmap_tensors
        self.loss_ops = []
        self.loss_op = 0.0
        for i, heatmap_tensor in heatmap_inferred_tensors:
            self.loss_ops.append(tf.losses.mean_squared_error(labels=heatmap_groundtruth_tensor, 
                                                              predictions=heatmap_tensor)
            self.loss_op = self.loss_op + self.loss_ops[-1]
        self.learning_rate=learning_rate
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_op)
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)
        self.saver = tf.train.Saver()
        return
    
    def predict(self, x):
        y = self.sess.run(self.y, feed_dict={self.x : x})
        return y

    def fit(self, x, t):
        _, loss = self.sess.run([self.train_op, self.loss_op], {self.x: x, self.target: t})

    def save(self, path: str):
        self.saver.save(self.sess, path)

    def load(self, path: str):
        self.saver.restore(self.sess, path)
        
                                    
                                    
                                   
    

with tf.Session(target='', graph=graph0, config=config) as sess:
    image_tensor = tf.placeholder(dtype=tf.uint8, shape=[None, None, None, 3])
    x1 = (tf.cast(x=image_tensor, dtype=tf.float32) * 2.0 / 255.0) - 1.0
    x2 = tf.image.resize_images(x1, size=(256, 256))
    heatmap_inferred_tensors = model.fan(x=x2, num_modules=1)
    heatmap_inferred_t1_tensor = heatmap_inferred_tensors[0]
    heatmap_groundtruth_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 68])
    loss_op = tf.losses.mean_squared_error(labels=heatmap_groundtruth_tensor, predictions=heatmap_inferred_t1_tensor)
    train_op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss_op)
    
    

    
    n_epoch = 200
    n_steps = len(data_iterator)

    for i_epoch in range(n_epoch):
        loss = 0.0
        for i_step, (image_sample, heatmap_sample) in enumerate(data_iterator):
            loss_i, _ = sess.run([loss_op, train_op], 
                                  feed_dict={image_tensor:[image_sample],
                                             heatmap_groundtruth_tensor:[heatmap_sample]})
            loss += loss_i
        print('epoch:', i_epoch, 'loss:', loss / n_steps)                                   
                                    
        


def parse_arguments():
    parser = argparse_parse
    return
                                    



def main(dump_dir):
    load_dataset_from_json_with_heatmap(dump_dir)
    






if __name__ == '__main__':
    args = parse_arguments
    main()
