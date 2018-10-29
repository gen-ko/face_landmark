import numpy as np
import tensorflow as tf
from face_landmark import preprocess
from face_detection import face_detector

def get_tf_dataset(record_path, batch_size=16, detector=None, eval_mode=False):
    def _parse_map_stage1(record):
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
        return [image_decoded, landmark]


    def _parse_map_stage2(image, landmark, use_original=False):
        if use_original:
            return image, preprocess.pts2heatmap(landmark, sigma=1, singular=eval_mode)

        image, landmark = preprocess.random_flip_lr(image, landmark)
        image, landmark = preprocess.random_scale(image, landmark)
        image, landmark = preprocess.random_rotate(image, landmark)
        try:
            bbx = detector.predict_1x1_raise(image, threshold=0.87)
            image, landmark = preprocess.train_preprocess(image, landmark, bbx)
        except face_detector.DetectionError:
            pass
        heatmap = preprocess.pts2heatmap(landmark, sigma=1, singular=eval_mode)
        return image, heatmap


    def _parse_map_stage3(image, landmark):
        '''
        Recover shape since rank and shape info are lost in py_func (stage2)
        '''
        image.set_shape([None, None, 3])
        x1 = (tf.cast(x=image, dtype=tf.float32) * 2.0 / 255.0) - 1.0
        x2 = tf.image.resize_images(x1, size=(256, 256))
        landmark.set_shape([64, 64, 68])
        return x2, landmark


    filenames = [record_path]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(_parse_map_stage1, num_parallel_calls=32)

    dataset = dataset.map(lambda img, ldm: tuple(tf.py_func(lambda x, y: _parse_map_stage2(x, y, False), inp=[img, ldm], Tout=[tf.uint8, tf.float32])))
    dataset = dataset.map(_parse_map_stage3, num_parallel_calls=32)
    dataset = dataset.batch(batch_size)
    return dataset

def train_input_fn(record_path, batch_size=16, sigma=1, detector=None):
    def _parse_map_stage1(record):
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
        return [image_decoded, landmark]

    def _parse_map_stage2(image, landmark):
        if detector is None:
            return image, preprocess.pts2heatmap(landmark, sigma=sigma, singular=False)

        image, landmark = preprocess.random_flip_lr(image, landmark)
        image, landmark = preprocess.random_scale(image, landmark)
        image, landmark = preprocess.random_rotate(image, landmark)
        try:
            bbx = detector.predict_1x1_raise(image, threshold=0.87)
            image, landmark = preprocess.train_preprocess(image, landmark, bbx)
        except face_detector.DetectionError:
            pass
        heatmap = preprocess.pts2heatmap(landmark, sigma=sigma, singular=False)
        return image, heatmap

    def _parse_map_stage3(image, landmark):
        '''
        Recover shape since rank and shape info are lost in py_func (stage2)
        '''
        image.set_shape([None, None, 3])
        x1 = (tf.cast(x=image, dtype=tf.float32) * 2.0 / 255.0) - 1.0
        x2 = tf.image.resize_images(x1, size=(256, 256))
        landmark.set_shape([64, 64, 68])
        return x2, landmark

    filenames = [record_path]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(_parse_map_stage1, num_parallel_calls=batch_size)
    dataset = dataset.map(lambda img, ldm: tuple(tf.py_func(lambda x, y: _parse_map_stage2(x, y), inp=[img, ldm], Tout=[tf.uint8, tf.float32])), num_parallel_calls=batch_size)
    dataset = dataset.map(_parse_map_stage3, num_parallel_calls=batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=1)
    return dataset

def eval_input_fn(record_path, batch_size=16, detector=None):
    def _parse_map_stage1(record):
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
        return [image_decoded, landmark]

    def _parse_map_stage2(image, landmark):
        if detector is None:
            return image, preprocess.pts2heatmap(landmark, sigma=1, singular=True)
        try:
            bbx = detector.predict_1x1_raise(image, threshold=0.87)
            image, landmark = preprocess.train_preprocess(image, landmark, bbx)
        except face_detector.DetectionError:
            pass
        heatmap = preprocess.pts2heatmap(landmark, sigma=1, singular=True)
        return image, heatmap

    def _parse_map_stage3(image, landmark):
        '''
        Recover shape since rank and shape info are lost in py_func (stage2)
        '''
        image.set_shape([None, None, 3])
        x1 = (tf.cast(x=image, dtype=tf.float32) * 2.0 / 255.0) - 1.0
        x2 = tf.image.resize_images(x1, size=(256, 256))
        landmark.set_shape([64, 64, 68])
        return x2, landmark

    filenames = [record_path]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(_parse_map_stage1, num_parallel_calls=batch_size)
    dataset = dataset.map(lambda img, ldm: tuple(tf.py_func(lambda x, y: _parse_map_stage2(x, y), inp=[img, ldm], Tout=[tf.uint8, tf.float32])), num_parallel_calls=batch_size)
    dataset = dataset.map(_parse_map_stage3, num_parallel_calls=batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=1)
    return dataset
