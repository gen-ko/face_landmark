import math
import numpy as np
import scipy
import tensorflow as tf
from PIL import Image
from skimage.transform import resize
from skimage.filters import gaussian

def extend_bbx_stage1(bbx):
    '''
    extend the bbx by 0.2x bbx_width left,
                      0.2x bbx_width right,
                      0.3x bbx_height down
    '''
    ymin, xmin, ymax, xmax = bbx
    w = xmax - xmin
    h = ymax - ymin
    ymin_new = ymin
    ymax_new = ymax + h * 0.3
    xmin_new = xmin - w * 0.2
    xmax_new = xmax + w * 0.2
    return np.array([ymin_new, xmin_new, ymax_new, xmax_new])

def flip_landmark_lr(pts):
    index_new = [16, 15, 14, 13, 12, 11, 10, 9,
                 8,
                 7, 6, 5, 4, 3, 2, 1, 0,
                 26, 25, 24, 23, 22,
                 21, 20, 19, 18, 17,
                 27, 28, 29, 30,
                 35, 34,
                 33,
                 32, 31,
                 45, 44, 43, 42,
                 47, 46,
                 39, 38, 37, 36,
                 41, 40,
                 54, 53, 52,
                 51,
                 50, 49, 48,
                 59, 58,
                 57,
                 56, 55,
                 64, 63,
                 62,
                 61, 60,
                 67,
                 66,
                 65]
    pts = pts[index_new]
    pts[:, 0] = 1.0 - pts[:, 0]
    return pts

def flip_lr(image, pts):
    image = np.flip(image, axis=1)
    pts = flip_landmark_lr(pts)
    return image, pts

def random_flip_lr(image, pts, probe=0.5):
    if np.random.uniform() < probe:
        image, pts = flip_lr(image, pts)
    return image, pts

def rotate_landmark(pts, angle):
    '''
    [x'] =  [1   ] [cos0  -sin0] [1   ] [x]
    [y']    [  -1] [sin0   cos0] [  -1] [y]
    '''
    pts = pts - 0.5
    angle = math.radians(angle)
    rotate_matrix = np.array([[math.cos(angle), math.sin(angle)],
                              [-math.sin(angle),math.cos(angle)]])
    pts = (np.matmul(rotate_matrix, pts.T)).T
    pts = pts + 0.5
    return pts

def rotate(image, pts, angle):
    image = Image.fromarray(image)
    image = image.rotate(angle, resample=Image.BILINEAR)
    pts = rotate_landmark(pts, angle)
    return np.array(image), pts

def random_rotate(image, pts, limit=50.0, probe=0.5):
    if np.random.uniform() < probe:
        angle = np.random.uniform(low=-limit, high=limit)
        image, pts = rotate(image, pts, angle)
    return image, pts

def scale(image, pts, h_ratio, w_ratio):
    h, w, _ = image.shape
    h = int(h * h_ratio)
    w = int(w * w_ratio)
    image = Image.fromarray(image)
    image = image.resize((h,w), resample=Image.BILINEAR)
    return np.array(image), pts

def random_scale(image, pts, sup=1.2, inf=0.8, probe=0.5):
    if np.random.uniform() < probe:
        h_ratio = np.random.uniform(low=inf, high=sup)
        w_ratio = np.random.uniform(low=inf, high=sup)
        image, pts = scale(image, pts, h_ratio, w_ratio)
    return image, pts

def extend_bbx_stage2(bbx):
    '''
    extend the bbx to make it 1:1 aspect ratio
    '''
    ymin, xmin, ymax, xmax = bbx
    w = xmax - xmin
    h = ymax - ymin
    delta = (h - w) * 0.5
    if delta > 0:
        xmin = xmin - delta
        xmax = xmax + delta
    else:
        delta = -delta
        ymin = ymin - delta
        ymax = ymax + delta
    return np.array([ymin, xmin, ymax, xmax])

def extend_bbx(bbx):
    bbx = extend_bbx_stage1(bbx)
    bbx = extend_bbx_stage2(bbx)
    return bbx

def extend_bbx_adaptive(bbx):
    '''
    returns the extended bbx, may containes negative value or values greater than 1
    -----------           ------------
    |         |           |          |
    |   ---   |           |  ------  |
    |   | |   |    -->    |  |    |  |
    |   ---   |           |  |    |  |
    |         |           |  ------  |
    -----------           ------------
    bbx: (4, ) array with dtype np.float32, range from [0, 1)
    '''
    ymin, xmin, ymax, xmax = bbx
    w = xmax - xmin
    h = ymax - ymin
    aspect_ratio = h / w
    if aspect_ratio > 1.0 and aspect_ratio < 1.7:
        # clip is necessary to avoid huge padding matrix
        delta_w = np.clip(((aspect_ratio - 1) / (2 - aspect_ratio) * w), 0.0, w)
    else:
        delta_w = 0.0
    xmin_new = xmin - delta_w
    xmax_new = xmax + delta_w
    ymin_new = ymin
    ymax_new = ymax + delta_w * aspect_ratio
    tmp = np.array([ymin_new, xmin_new, ymax_new, xmax_new], dtype=np.float32)
    return tmp


def get_extend_matrix(bbx):
    '''
    bbx: the extended bounding box
    returns a 3x3 matrix indicating the transform of the extending the image
    '''
    ymin, xmin, ymax, xmax = bbx
    exmin = max(0.0, 0.0-xmin)
    exmax = max(0.0, xmax-1.0)
    eymin = max(0.0, 0.0-ymin)
    eymax = max(0.0, ymax-1.0)
    ex = exmin+exmax
    fx = ex+1.0
    ey = eymin+eymax
    fy = ey+1.0
    tmp = np.array([[1.0/fx, 0.0, exmin/fx],
                    [0.0, 1.0/fy,eymin/fy],
                    [0.0,0.0,1.0]])
    return tmp


def extend_image(image, extend_matrix):
    '''
    image must be in HWC format, does not care about the number of C
    '''
    exmin = extend_matrix[0,2] / (extend_matrix[0,0])
    exmax = 1.0 / (extend_matrix[0, 0]) - exmin - 1.0
    eymin = extend_matrix[1,2] / (extend_matrix[1,1])
    eymax = 1.0 / (extend_matrix[1, 1]) - eymin - 1.0
    #print('extend_image:', exmin, exmax, eymin, eymax)
    height =  image.shape[0]
    width = image.shape[1]
    exmin = int(width * exmin)
    exmax = int(width * exmax)
    eymin = int(height * eymin)
    eymax = int(height * eymax)
    image = np.pad(image, pad_width=((eymin, eymax), (exmin, exmax), (0, 0)), mode='constant', constant_values=0)
    return image


def get_crop_matrix(bbx):
    esp = 1e-7
    ymin, xmin, ymax, xmax = bbx
    cwidth = xmax - xmin
    cheight = ymax - ymin
    tmp = np.array([[1.0/(cwidth),0.0,-xmin/(cwidth)],
                    [0.0,1.0/(cheight),-ymin/(cheight)],
                    [0.0, 0.0, 1.0]])
    #print('get crop mat', ymin, xmin, ymax, xmax, cwidth, cheight)
    #print('crop mat', tmp)
    return tmp


def crop_image(image, crop_matrix):
    '''
    does not care about image dtype,
    does not care about image channel
    the image dimension must be 2 (HW) or 3 (HWC)
    '''
    esp = 1e-7
    xmin = -crop_matrix[0, 2] / (crop_matrix[0, 0])
    xmax = 1.0/(crop_matrix[0,0]) + xmin
    ymin = -crop_matrix[1, 2] / (crop_matrix[1, 1])
    ymax = 1.0/(crop_matrix[1,1]) + ymin
    #print('crop_image', xmin, xmax, ymin, ymax)
    #print('crop mat', crop_matrix)
    height = image.shape[0]
    width = image.shape[1]
    xmin = int(width * xmin)
    xmax = int(width * xmax)
    ymin = int(height * ymin)
    ymax = int(height * ymax)
    image = image[ymin:ymax, xmin:xmax]
    return image


def transform_pts(pts, transform_matrix):
    '''
    pts: (M, 2) matrix
         the first column indicates the x coordinates,
         the second column indatecs the y coordinates
    transform_matrix: (3, 3) matrix

    [ x'] = [ T00, T01, T02 ] [x]
    [ y']   [ T10, T11, T12 ] [y]
    [ 1 ]   [ T20, T21, T22 ] [1]
    '''
    pts = np.pad(pts, pad_width=((0,0), (0,1)), mode='constant', constant_values=1.0)
    pts = np.matmul(pts, transform_matrix.T)
    pts = pts[:, :2]
    return pts


def train_preprocess(image, landmark, bbx):
    image, trans_matrix = infer_preprocess(image, bbx)
    landmark = transform_pts(landmark, trans_matrix)
    return image, landmark


def infer_preprocess(image, bbx):
    bbx = extend_bbx(bbx)
    extend_matrix = get_extend_matrix(bbx)

    ymin, xmin, ymax, xmax = bbx
    bbx_pts = np.array([[xmin, ymin],
                        [xmax, ymax]]).astype(np.float32)

    bbx_pts = transform_pts(bbx_pts, extend_matrix)
    xmin, ymin = bbx_pts[0]
    xmax, ymax = bbx_pts[1]
    bbx_crop = np.array([ymin, xmin, ymax, xmax])
    crop_matrix = get_crop_matrix(bbx_crop)

    image = extend_image(image, extend_matrix)
    image = crop_image(image, crop_matrix)
    trans_matrix = np.matmul(crop_matrix, extend_matrix)

    return image, trans_matrix

def infer_postprocess(landmark, trans_matrix):
    inv_matrix = np.linalg.inv(trans_matrix)
    landmark = transform_pts(landmark, inv_matrix)
    landmark = np.clip(landmark, 0.0, 1.0)
    return landmark


def pts2heatmap(pts, h=64, w=64, sigma=1, singular=False):
    '''
    :pts: 2D array, [[x0, y0], [x1, y1], ..., [x(c-1), y(c-1)]]  x, y in range [0, 1)
    '''
    eps = 1e-7
    c = pts.shape[0]
    heatmap = np.zeros((h, w, c), dtype=np.float32)
    pts = np.clip(pts, 0.0, 1.0-eps)

    index_ys = (pts[:, 1]*h).astype(int)
    index_xs = (pts[:, 0]*w).astype(int)
    index_cs = np.arange(c)
    heatmap[index_ys, index_xs, index_cs] = 1.0
    if singular:
        return heatmap
    
    heatmap = gaussian(image=heatmap, 
                       sigma=sigma, 
                       output=None, 
                       mode='constant', 
                       cval=0.0, 
                       multichannel=True, 
                       preserve_range=False, 
                       truncate=4.0)
    # an alternative way, the gaussian kernel are not truncated
    # heatmap = scipy.ndimage.gaussian_filter(heatmap, sigma=sigma, mode='constant')
    max_itensity = heatmap.max(axis=(0,1))
    min_itensity = heatmap.min(axis=(0,1))
    scale = 2.0 / (max_itensity - min_itensity)
    mean = (max_itensity - min_itensity) / 2.0
    heatmap = (heatmap - mean) * scale
    return heatmap


def heatmap2pts(heatmap):
    '''
    heatmap: NHWC tensor
    
    NOTE: This is an equivalent implementation of heatmap2pts
    '''
    shape = tf.shape(heatmap)
    h = shape[1]
    w = shape[2]
    c = shape[3]
    heatmap = tf.reshape(heatmap, [-1, h * w, c])
    indice = tf.cast(tf.argmax(heatmap, axis=1), dtype=tf.int32)
    rows = tf.cast(indice / w, dtype=tf.int32)
    cols = indice % w
    ys = (tf.cast(rows, dtype=tf.float32) + 0.5) / tf.cast(h, tf.float32)
    xs = (tf.cast(cols, dtype=tf.float32) + 0.5) / tf.cast(w, tf.float32)
    pts = tf.stack([xs, ys], axis=2)
    return pts



