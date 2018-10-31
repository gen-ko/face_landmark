#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from face_landmark import preprocess
from face_detection import face_detector

def _parse_map_stage1(record):
    """
    parse tfrecord into image and landmark pts

    return valuess:
    image_decoded: 3-D Tensor, tf.uint8, HWC format, RGB channels range from [0, 255]
    landmark: 2-D Tensor, tf.float32
              specifically, for 68-point landmark annotations, the shape is [68, 2]
              typically the value range is [0, 1], but it is possible to exceed this range
              for each landmark point, the 0th value is the x coordinate, the 1st value is the y coordinate
    """
    feature_dict = {}
    #feature_dict['image/width'] = tf.FixedLenFeature([], dtype=tf.int64)
    #feature_dict['image/height'] = tf.FixedLenFeature([], dtype=tf.int64)
    feature_dict['image/encoded'] = tf.FixedLenFeature([], dtype=tf.string)
    #feature_dict['image/key/sha256'] = tf.FixedLenFeature([], dtype=tf.string)
    feature_dict['landmark/flatten'] = tf.VarLenFeature(dtype=tf.float32)
    #feature_dict['landmark/shape'] = tf.FixedLenFeature([], dtype=tf.int64)
    example = tf.parse_single_example(record, features=feature_dict)
    #image_width = example['image/width']
    #image_height = example['image/height']
    image_encoded = example['image/encoded']
    image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)
    #image_id = example['image/key/sha256']
    landmark = example['landmark/flatten']
    landmark = tf.sparse_tensor_to_dense(landmark)
    landmark = tf.reshape(landmark, [68, 2])
    return image_decoded, landmark

def _parse_map_stage2(image, landmark, sigma=5, detector=None, augmentation=None, preserve_bbx=False, generate_heatmap=False):
    """works outside tf graph
    image: 3-D np.ndarray, HWC, uint8, RGB, range from [0, 255]
    landmark: 2-D np.ndarray, (68, 2), range from [0, 1] (exceeding allowed)
    """
    if detector is None:
        detector = lambda x: np.array([0.0, 0.0, 1.0, 1.0])

    bbx = detector(image)
    
    image, landmark = preprocess.train_preprocess(image, landmark, bbx)
    if augmentation is not None:
        image, landmark = augmentation(image, landmark)
    
    if generate_heatmap:
        landmark = preprocess.pts2heatmap(landmark, h=256, w=256, sigma=sigma, singular=False).astype(np.float32)

    landmark = landmark.astype(np.float32)
    
    if preserve_bbx:
        bbx = bbx.astype(np.float32)
        return image, landmark, bbx
    else:
        return image, landmark


def _parse_map_stage3(image, landmark):
    '''
    Recover shape since rank and shape info are lost in py_func (stage2)
    '''
    image.set_shape([None, None, 3])
    # without normalization
    image = (tf.cast(x=image, dtype=tf.float32) * 1.0 / 255.0)
    image = tf.image.resize_images(image, size=(256, 256))
    landmark.set_shape([256, 256, 68])
    return image, landmark

def _parse_map_stage4(image, *args):
    image = image * 2 - 1.0
    res = [image]
    for arg in args:
        res.append(arg)
    return res


def _augmentation_tf(image):
    """
    image: 4-D Tensor with shape (batch_size, H, W, C)
    Augment image while keep the landmark invariant
    """
    image = tf.image.random_hue(image, 0.2)    
    image = tf.image.random_brightness(image, 0.0)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image
    
def _augmentation_np(image, landmark):
    image, landmark = preprocess.random_flip_lr(image, landmark)
    image, landmark = preprocess.random_scale(image, landmark)
    image, landmark = preprocess.random_rotate(image, landmark)
    image, landmark = preprocess.random_occulusion(image, landmark)
    return image, landmark
    
def input_fn(record_path, batch_size=16, sigma=1, detector=None, is_eval=False):
    filenames = [record_path]
    dataset = tf.data.TFRecordDataset(filenames)
    if not is_eval:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(_parse_map_stage1, num_parallel_calls=batch_size)
    if not is_eval:
        dataset = dataset.map(lambda img, ldm: tuple(tf.py_func(lambda x, y: _parse_map_stage2(x, y, 
                                                        sigma=sigma,
                                                        detector=detector,
                                                        augmentation=_augmentation_np,
                                                        preserve_bbx=False, generate_heatmap=True), inp=[img, ldm], Tout=[tf.uint8, tf.float32])), num_parallel_calls=batch_size)
    else:
        dataset = dataset.map(
            lambda img, ldm: tuple(tf.py_func(lambda x, y: _parse_map_stage2(x, y, 
                sigma=sigma,
                detector=detector,
                augmentation=None,
                preserve_bbx=True, generate_heatmap=False), inp=[img, ldm], Tout=[tf.uint8, tf.float32, tf.float32])), num_parallel_calls=batch_size)
    if not is_eval:
        dataset = dataset.map(_parse_map_stage3, num_parallel_calls=batch_size)
    else:
        dataset = dataset.map(lambda img, ldm, bbx: (*_parse_map_stage3(img, ldm), bbx), num_parallel_calls=batch_size)
    dataset = dataset.batch(batch_size)
    if not is_eval:
        dataset = dataset.map(lambda img, ldm: (_augmentation_tf(img), ldm))
    dataset = dataset.map(_parse_map_stage4)
    dataset = dataset.prefetch(buffer_size=1)
    return dataset  

