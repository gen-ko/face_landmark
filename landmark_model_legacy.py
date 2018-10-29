import numpy as np
import tensorflow as tf



'''
class LandmarkModel(object):
    def __init__(self, model_dir: str):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        graph = tf.Graph()
        self.sess = tf.Session(config=config, graph=graph)
        with graph.as_default():
            tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], model_dir)                                      
            self.x = tf.get_default_graph().get_tensor_by_name("x:0")
            self.y = tf.get_default_graph().get_tensor_by_name("y:0")
            init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        return

    def predict_batch(self, x):
        '''
        x is the cropped image, either cropped by a bounding box or normalized extended bounding box
        x is in NHWC format
        '''
        heatmap =  self.sess.run(self.y, {self.x:x})
        


    def predict(self, x):
        heatmap = self.sess.run(self.y, {self.x: [x])[0]

        x_flipped = 

        
    def postprocess(self, heatmap):
'''
