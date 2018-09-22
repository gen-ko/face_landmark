import re
import os
import json
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imsave
from torch.utils.serialization import load_lua
from tqdm import tqdm
from utils import pipeline
import cv2
import tensorflow as tf
from face_detection import face_detector
import PIL
# Currently working with LF3D dataset
ANNOTATION_PATTERN = re.compile(pattern='([ \\S]+)_pts.t7')


def load_annotation(annotation_path):
    '''
    :return: 68x2 positions of the annotation
    '''
    annotations = load_lua(annotation_path)
    anno_2d = annotations[0].numpy()
    anno_3d = annotations[1].numpy()
    return anno_2d, anno_3d

def load_annotation_batch(annotation_path_list):
    anno_2ds = []
    anno_3ds = []
    for annotation_path in annotation_path_list:
        anno_2d, anno_3d = load_annotation(annotation_path)
        anno_2ds.append(anno_2d)
        anno_3ds.append(anno_3d)
    anno_2ds = np.array(anno_2ds)
    anno_3ds = np.array(anno_3ds)
    return anno_2ds, anno_3ds

def load_image_batch(image_path_list):
    images = []
    for image_path in image_path_list:
        # HxWx3
        image = imread(image_path, flatten=False, mode='RGB')
        images.append(image)
    images = np.array(images)
    return images


# mapping from annotation to image file
def LF3D_name_mapping(annotation_name):
    match = ANNOTATION_PATTERN.fullmatch(annotation_name)
    if match is not None:
        return match.groups()[0] + '.jpg'
    else:
        return None

def heatmap_name_mapping(annotation_name):
    match = ANNOTATION_PATTERN.fullmatch(annotation_name)
    if match is not None:
        return match.groups()[0] + '_hm.npy'
    else:
        return None

def load_data_filenames(annotation_root_dir, image_root_dir):
    anno_filenames = pipeline.search_files(dir= annotation_root_dir, pattern='([ \\S]+)_pts.t7')
    img_filenames = []
    for anno_filename in anno_filenames:
        dirname, basename = os.path.split(anno_filename)
        relname = os.path.relpath(dirname, annotation_root_dir)
        img_basename = LF3D_name_mapping(basename)
        img_filename = os.path.join(image_root_dir, relname, img_basename)
        img_filenames.append(img_filename)
    return anno_filenames, img_filenames


def generate_heatmap_filenames(annotation_filenames,annotation_root_dir, heatmap_root_dir):
    img_filenames = []
    for anno_filename in annotation_filenames:
        dirname, basename = os.path.split(anno_filename)
        relname = os.path.relpath(dirname, annotation_root_dir)
        img_basename = heatmap_name_mapping(basename)
        img_filename = os.path.join(heatmap_root_dir, relname, img_basename)
        img_filenames.append(img_filename)
    return img_filenames

def load_data(annotation_root_dir, image_root_dir):
    anno_filenames, img_filenames = load_data_filenames(annotation_root_dir, image_root_dir)
    anno_2ds, anno_3ds = load_annotation_batch(anno_filenames)
    images = load_image_batch(img_filenames)
    return anno_2ds, anno_3ds, images

def create_dataset(annotation_root_dir, image_root_dir, dump_dir):
    anno_filenames, img_filenames = load_data_filenames(annotation_root_dir, image_root_dir)
    with open(os.path.join(dump_dir, 'anno_filenames.json'), 'w+') as fid:
        json.dump(anno_filenames, fid)
    with open(os.path.join(dump_dir, 'img_filenames.json'), 'w+') as fid:
        json.dump(img_filenames, fid)
    return

def load_dataset_from_json(dump_dir):
    with open(os.path.join(dump_dir, 'anno_filenames.json'), 'r') as fid:
        anno_filenames = json.load(fid)
    with open(os.path.join(dump_dir, 'img_filenames.json'), 'r') as fid:
        imgs_filenames = json.load(fid)
    return anno_filenames, imgs_filenames

def load_dataset_from_json_with_heatmap(dump_dir):
    with open(os.path.join(dump_dir, 'heatmap_filenames.json'), 'r') as fid:
        heatmap_filenames = json.load(fid)
    with open(os.path.join(dump_dir, 'img_filenames.json'), 'r') as fid:
        imgs_filenames = json.load(fid)
    return heatmap_filenames, imgs_filenames

def generate_heatmap(pts):
    # The desired Height and Width of the heatmaps
    # NOTE: pts must be normalized to [0, 1)
    h = 64
    w = 64
    c = pts.shape[0]
    assert pts.shape == (68, 2)
    #assert pts.dtype == np.float32
    heatmap = np.zeros((h, w, c), dtype=np.float32)
    pts = np.clip(pts, 0.0, 1-1e-7) 
    # NOTE: For int(64 * np.float32(1 - 1e-8)) = 64, yet int(64 * np.float32(1 - 1e-7)) = 63 
    for i in range(c):
        # the first  [0] column of pts is x
        # the second [1] column of pts is y 
        index_y = int(pts[i][1] * h)
        index_x = int(pts[i][0] * w)
        try:
            heatmap[index_y, index_x, i] = 1.0
        except IndexError:
            print('index_y', index_y)
            print('index_x', index_x)
            print('i', i)
            raise IndexError
    heatmap = cv2.GaussianBlur(heatmap, (h + (h % 2) - 1, w + (w % 2) - 1), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_CONSTANT)
    max_itensity = heatmap.max(axis=(0,1))
    min_itensity = heatmap.min(axis=(0,1))
    scale = 2.0 / (max_itensity - min_itensity)
    mean = (max_itensity + min_itensity) / 2.0
    heatmap = (heatmap - mean) * scale
    heatmap = np.array(heatmap)
    return heatmap

def create_heatmap_dataset(dump_dir, annotation_root_dir, heatmap_root_dir):
    '''
    first the json file for the annotation filenames should be generated,
    then run this script to generate corrsponding heatmap npys
    meanwhile, 68 samples of single channels will be saved as jpeg examples for visualization
    '''
    anno_filenames, img_filenames = load_dataset_from_json(dump_dir)
    heatmap_filenames = generate_heatmap_filenames(anno_filenames, annotation_root_dir, heatmap_root_dir)
    sample_heatmap = np.random.randint(low=0, high=len(anno_filenames), size=int(0.05*len(anno_filenames)+min(10, len(anno_filenames))), dtype=int)
    # dump the filenames of generated heatmap into a json file
    with open(os.path.join(dump_dir, 'heatmap_filenames.json'), 'w+') as fid:
        json.dump(heatmap_filenames, fid)
        
    for i, anno_filename in enumerate(anno_filenames):
        image = imread(img_filenames[i], flatten=False, mode='RGB')
        h = image.shape[0]
        w = image.shape[1]
        anno_2d, anno_3d = load_annotation(anno_filename)
        anno_3d = anno_3d.astype(np.float32)
        anno_3d[:,0] = np.float32(anno_3d[:,0]/w)
        anno_3d[:,1] = np.float32(anno_3d[:,1]/h)
        heatmap = generate_heatmap(anno_3d)
        #print(heatmap.max())
        #print(heatmap.min())
        dirname, basename = os.path.split(heatmap_filenames[i])
        os.makedirs(dirname, exist_ok=True)
        np.save(heatmap_filenames[i], heatmap)
        if i in sample_heatmap:
            # index = np.where(sample_heatmap == i)
            channel = np.random.randint(low=0, high=68)
            img = heatmap[:,:,channel]
            img = (img + 1) * 255.0 / 2.0
            img = img.astype(np.uint8)
            #print('max', img.max(), '\n')
            #print('min', img.min(), '\n')
            imsave(os.path.join(heatmap_root_dir, 'samples', 'sample_hm'+str(i)+'.png'), img)
    return 


class DatasetIterator(object):
    def __init__(self, dump_dir):
        anno_filenames, imgs_filenames = load_dataset_from_json(dump_dir)
        self.anno_filenames = anno_filenames
        self.imgs_filenames = imgs_filenames
        self.cnt = 0
        return

    def __iter__(self):
        self.cnt = 0
        return self

    def __next__(self):
        if self.cnt >= len(self.anno_filenames):
            raise StopIteration
        else:
            anno_path = self.anno_filenames[self.cnt]
            img_path = self.imgs_filenames[self.cnt]
            anno_2d, anno_3d = load_annotation(anno_path)
            img = imread(img_path, flatten=False, mode='RGB')
            self.cnt += 1
            return (anno_2d, anno_3d), img

    def __len__(self):
        return len(self.anno_filenames)
    
class DatasetIteratorMixed(object):
    def __init__(self, dump_dir):
        anno_filenames, imgs_filenames = load_dataset_from_json(dump_dir)
        self.anno_filenames = anno_filenames
        self.imgs_filenames = imgs_filenames
        self.cnt = 0
        return

    def __iter__(self):
        self.cnt = 0
        return self

    def __next__(self):
        if self.cnt >= len(self.anno_filenames):
            raise StopIteration
        else:
            anno_path = self.anno_filenames[self.cnt]
            img_path = self.imgs_filenames[self.cnt]
            _, anno_3d = load_annotation(anno_path)
            self.cnt += 1
            return img_path, anno_3d

    def __len__(self):
        return len(self.anno_filenames)
    
class DatasetIteratorWithHeatmap(object):
    def __init__(self, dump_dir):
        heatmap_filenames, image_filenames = load_dataset_from_json_with_heatmap(dump_dir=dump_dir)
        self.heatmap_filenames = heatmap_filenames
        self.image_filenames = image_filenames
        self.cnt = 0
        return

    def __iter__(self):
        self.cnt = 0
        return self

    def __next__(self):
        if self.cnt >= len(self.heatmap_filenames):
            raise StopIteration
        else:
            heatmap_path = self.heatmap_filenames[self.cnt]
            image_path = self.image_filenames[self.cnt]
            heatmap = np.load(heatmap_path)
            image = imread(image_path, flatten=False, mode='RGB')
            self.cnt += 1
            return image, heatmap

    def __len__(self):
        return len(self.heatmap_filenames)
    
class DatasetIteratorWithHeatmapFilenamesOnly(object):
    def __init__(self, dump_dir):
        heatmap_filenames, image_filenames = load_dataset_from_json_with_heatmap(dump_dir=dump_dir)
        self.heatmap_filenames = heatmap_filenames
        self.image_filenames = image_filenames
        self.cnt = 0
        return

    def __iter__(self):
        self.cnt = 0
        return self

    def __next__(self):
        if self.cnt >= len(self.heatmap_filenames):
            raise StopIteration
        else:
            heatmap_path = self.heatmap_filenames[self.cnt]
            image_path = self.image_filenames[self.cnt]
            self.cnt += 1
            return image_path, heatmap_path

    def __len__(self):
        return len(self.heatmap_filenames)
    

    
class Dataset(object):
    def __init__(self, new_dim=256):
        self.detector = face_detector.FaceDetector()
        self.new_dim = new_dim
            
    def auto_crop(self, image, landmark):
        try:
            bbx = self.detector.predict_1x1(image)
        except face_detector.DetectionError:
            bbx = np.array([0.0, 0.0, 1.0, 1.0])
        height_raw, width_raw, _ = image.shape
        
        # landmarks 
        landmark_xmin = landmark[:,0].min()
        landmark_xmax = landmark[:,0].max()
        landmark_ymin = landmark[:,1].min()
        landmark_ymax = landmark[:,1].max()
        
        self.landmark_xmin_stat.append((bbx[1] - landmark_xmin) / (bbx[3] - bbx[1]))
        self.landmark_xmax_stat.append((bbx[3] - landmark_xmax) / (bbx[3] - bbx[1]))
        self.landmark_ymin_stat.append((bbx[0] - landmark_ymin) / (bbx[2] - bbx[0]))
        self.landmark_ymax_stat.append((bbx[2] - landmark_ymax) / (bbx[2] - bbx[0]))
        self.aspect_ratio_stat.append((bbx[2] - bbx[0]) / (bbx[3] - bbx[1]))
        
        crop_xmin = max(min(bbx[1], landmark_xmin), 0.0)
        crop_xmax = min(max(bbx[3], landmark_xmax), 1.0)
        crop_ymin = max(min(bbx[0], landmark_ymin), 0.0)
        crop_ymax = min(max(bbx[2], landmark_ymax), 1.0)
        
        crop_width = crop_xmax - crop_xmin
        crop_height = crop_ymax - crop_ymin
                
        # cropped origin  [0, 1)
        origin_x = bbx[1] 
        origin_y = bbx[0]
        
        # shift
        scale_x = 1.0 / crop_width
        scale_y = 1.0 / crop_height
        shift_x = - crop_xmin / crop_width
        shift_y = - crop_ymin / crop_height
        
        
        transform_matrix = np.array([[scale_x, 0, shift_x],
                                     [0, scale_y, shift_y],
                                     [0, 0, 1]]).astype(np.float32)
        landmark_e = np.ones([68, 3], dtype=np.float32)
        landmark_e[:, :2] = landmark
        landmark = np.matmul(landmark_e, transform_matrix.T)[:, :2]
        image = image[int(height_raw*crop_ymin):int(height_raw*crop_ymax), int(width_raw*crop_xmin):int(width_raw*crop_xmax)]
        return image, landmark
    
    def auto_crop_v2(self, image, landmark):
        '''
        this version of auto_crop does not require the info from landmark
        
        '''
        try:
            bbx = self.detector.predict_1x1(image)
        except face_detector.DetectionError:
            bbx = np.array([0.0, 0.0, 1.0, 1.0])
        height_raw, width_raw, _ = image.shape
               
        bbx_width = bbx[3] - bbx[1]
        bbx_height = bbx[2] - bbx[0]
        
        crop_xmin = max(bbx[1]-0.058 * bbx_width, 0.0)
        crop_xmax = min(bbx[3]+0.058 * bbx_width, 1.0)
        crop_ymax = min(bbx[2]+0.05  * bbx_height, 1.0)
        crop_ymin = min(bbx[0]+0.15  * bbx_height, crop_ymax - 0.1 * bbx_height)
        
        
        crop_width = crop_xmax - crop_xmin
        crop_height = crop_ymax - crop_ymin
                
        # cropped origin  [0, 1)
        origin_x = bbx[1] 
        origin_y = bbx[0]
        
        # shift
        scale_x = 1.0 / crop_width
        scale_y = 1.0 / crop_height
        shift_x = - crop_xmin / crop_width
        shift_y = - crop_ymin / crop_height
        
        
        transform_matrix = np.array([[scale_x, 0, shift_x],
                                     [0, scale_y, shift_y],
                                     [0, 0, 1]]).astype(np.float32)
        landmark_e = np.ones([68, 3], dtype=np.float32)
        landmark_e[:, :2] = landmark
        landmark = np.matmul(landmark_e, transform_matrix.T)[:, :2]
        landmark = np.clip(landmark, 0.0, 1.0)
        image = image[int(height_raw*crop_ymin):int(height_raw*crop_ymax), int(width_raw*crop_xmin):int(width_raw*crop_xmax)]
        return image, landmark
    
    
    def auto_crop_and_resize(self, image, landmark, target_size=256):
        image, landmark = self.auto_crop_v2(image, landmark)
        image = PIL.Image.fromarray(image)
        image = image.resize([target_size, target_size], resample=PIL.Image.BILINEAR)
        return image, landmark
        

        
    def get_tf_dataset(self, record_path, batch_size=16):
        def _parse_map_stage1(record):
            feature_dict = {}
            feature_dict['image/width'] = tf.FixedLenFeature([], dtype=tf.int64)
            feature_dict['image/height'] = tf.FixedLenFeature([], dtype=tf.int64)
            feature_dict['image/encoded'] = tf.FixedLenFeature([], dtype=tf.string)
            feature_dict['image/key/sha256'] = tf.FixedLenFeature([], dtype=tf.string)
            feature_dict['landmark/flatten'] = tf.VarLenFeature(dtype=tf.float32)
            example = tf.parse_single_example(record, features=feature_dict)

            image_width = example['image/width']
            image_height = example['image/height']
            image_encoded = example['image/encoded']
            image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)

            image_id = example['image/key/sha256']
            landmark_t1 = example['landmark/flatten']
            landmark_t2 = tf.sparse_tensor_to_dense(landmark_t1)
            landmark = tf.reshape(landmark_t2, [68, 2])
            landmark = landmark / tf.cast([image_width, image_height], dtype=tf.float32)
            return [image_decoded, landmark]

        #def _parse_map_stage2a(image, landmark):
            #return self.auto_cropper.auto_crop(auto_crop(image_landmark))

        def _parse_map_stage2(image, landmark):
            image, landmark = self.auto_crop_and_resize(image, landmark)
            return image, generate_heatmap(landmark)
                    

        def _parse_map_stage3(image, landmark):
            '''
            Recover shape since rank and shape info are lost in py_func (stage2) 
            '''
            image.set_shape([None, None, 3])
            x1 = (tf.cast(x=image, dtype=tf.float32) * 2.0 / 255.0) - 1.0
            #x2 = tf.image.resize_images(x1, size=(self.new_dim, self.new_dim))
            x2 = x1
            x2.set_shape([self.new_dim, self.new_dim, 3])
            landmark.set_shape([64, 64, 68])
            return x2, landmark


        filenames = [record_path]
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.map(_parse_map_stage1, num_parallel_calls=32)

        dataset = dataset.map(lambda img, ldm: tuple(tf.py_func(_parse_map_stage2, inp=[img, ldm], Tout=[tf.uint8, tf.float32])))
        dataset = dataset.map(_parse_map_stage3, num_parallel_calls=32)
        dataset = dataset.batch(batch_size)
        return dataset

    



