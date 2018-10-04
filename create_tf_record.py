import io
import tensorflow as tf
import hashlib
import PIL
import os
import numpy as np

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_example(image_filename, landmark):
    feature_dict = {}
    with tf.gfile.GFile(image_filename, 'rb') as fid:
        encoded_jpg = fid.read()
        #image = np.array(PIL.Image.open(fid))   # Use io to read from memory instead(which is much faster)
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = np.array(PIL.Image.open(encoded_jpg_io))
    key = hashlib.sha256(encoded_jpg).hexdigest()

    assert landmark.shape == (68, 2)
    landmark = landmark.flatten().tolist()

    feature_dict['image/height'] = _int64_feature(image.shape[0])
    feature_dict['image/width'] = _int64_feature(image.shape[1])
    feature_dict['image/channel'] = _int64_feature(image.shape[2])
    feature_dict['image/encoded'] = _bytes_feature(encoded_jpg)
    feature_dict['image/format'] = _bytes_feature('jpeg'.encode('utf8'))
    feature_dict['image/key/sha256'] = _bytes_feature(key.encode('utf8'))

    feature_dict['landmark/flatten'] = _float_list_feature(landmark)
    feature_dict['landmark/shape'] = _int64_list_feature([68, 2])
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example

def create_tf_record(iterator, dest_filename):
    if os.path.isfile(dest_filename):
        print('found dest filename:', dest_filename, 'deleting it...')
        os.remove(dest_filename)
        print('deleted.')
    writer = tf.python_io.TFRecordWriter(dest_filename)
    for pair in iterator:
        # definition of pair:
        # image_filename, landmark
        # image must be encoded as jpeg
        # landmark must be normalized to [0, 1) with dtype np.float32 to make it compatible with loading
        # it is recomended to shuffle the whole iterator before calling create_tf_record
        example = create_example(*pair)
        writer.write(example.SerializeToString())
    writer.close()
    return
