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
    bbx: 1D tensor [ymin, xmin, ymax, xmax]
    return: 1D tensor, with shape [4], dtype float32, [ymin, xmin, ymax, xmax]
    '''
    # [w, h]
    bbx = tf.reshape(bbx, [4, 1])
    wh = tf.matmul([[0.0, -1.0, 0.0, 1.0], [-1.0, 0.0, 1.0, 0.0]], bbx)
    delta = tf.matmul([[0.0, 0.0], [-0.2, 0], [0, 0.3], [0.2, 0]], wh)
    bbx = bbx + delta
    bbx = tf.reshape(bbx, [4])
    return bbx

def extend_bbx_stage1_batch(bbx):
    '''
    bbx: 2D tensor, shape [batch_size, 4]
    return: 2D tensor, with shape [batch_size, 4], dtype float32, [ymin, xmin, ymax, xmax]
    '''
    bbx = tf.reshape(bbx, [-1, 4, 1])
    transform = tf.convert_to_tensor([[[0.0, -1.0, 0.0, 1.0], [-1.0, 0.0, 1.0, 0.0]]])
    shape = tf.shape(bbx)
    batch_size = shape[0]
    transform = tf.manip.tile(transform, multiples=[batch_size, 1, 1])
    wh = tf.matmul(transform, bbx)
    transform = tf.convert_to_tensor([[[0.0, 0.0], [-0.2, 0], [0, 0.3], [0.2, 0]]])
    transform = tf.manip.tile(transform, multiples=[batch_size, 1, 1])
    delta = tf.matmul(transform, wh)
    bbx = bbx + delta
    bbx = tf.reshape(bbx, [batch_size, 4])
    return
    
    
def flip_landmark_lr(pts):
    '''
    pts: 2D tensor [m, 2], where m is the pts num of a landmark, typically 68,  2 is the x, y coordinate
    '''
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
    depth = len(index_new)
    one_hot = tf.one_hot(indices=index_new, depth=depth)
    pts = tf.matmul(one_hot, pts)
    delta_y1 = tf.matmul(pts, [[1.0,0.0],[0.0,-1.0]])
    const_y2 = np.repeat([[0, 1.0]], repeats=depth, axis=0)
    pts = const_y2 + pts
    return pts

def flip_lr(image, pts):
    '''
    image: 3-D Tensor of shape [height, width, channels]
    pts: 2-D Tensor of shape [m, 2]
    '''
    image = tf.image.flip_left_right(image)
    pts = flip_landmark_lr(pts)
    return image, pts

def random_flip_lr(image, pts, probe=0.5):
    '''
    image: 3-D Tensor of shape [height, width, channels]
    pts: 2-D Tensor of shape [m, 2]
    '''
    random_num = tf.random_uniform(shape=[])
    image, pts = tf.cond(
            pred=random_num < probe,
            true_fn= lambda: flip_lr(image, pts),
            false_fn=lambda: (image, pts))
    return image, pts


'''
No equivalent implementation needed for inference
def rotate_landmark(pts, angle):
    """
    [x'] =  [1   ] [cos0  -sin0] [1   ] [x]
    [y']    [  -1] [sin0   cos0] [  -1] [y]
    """
    pts = pts - 0.5
    angle = math.radians(angle)
    rotate_matrix = np.array([[math.cos(angle), math.sin(angle)],
                              [-math.sin(angle),math.cos(angle)]])
    pts = tf.matmul(pts, tf.transpose(rotate_matrix))
    pts = pts + 0.5
    return pts
'''
     
'''
No equivalent implementation needed for inference
def rotate(image, pts, angle):
    image = Image.fromarray(image)
    image = image.rotate(angle, resample=Image.BILINEAR)
    pts = rotate_landmark(pts, angle)
    return np.array(image), pts
'''
'''
No equivalent implementation needed for inference
def random_rotate(image, pts, limit=50.0, probe=0.5):
    if np.random.uniform() < probe:
        angle = np.random.uniform(low=-limit, high=limit)
        image, pts = rotate(image, pts, angle)
    return image, pts
'''
'''
No equivalent implementation needed for inference
def scale(image, pts, h_ratio, w_ratio):
    h, w, _ = image.shape
    h = int(h * h_ratio)
    w = int(w * w_ratio)
    image = Image.fromarray(image)
    image = image.resize((h,w), resample=Image.BILINEAR)
    return np.array(image), pts
'''
'''
No equivalent implementation needed for inference
def random_scale(image, pts, sup=1.2, inf=0.8, probe=0.5):
    if np.random.uniform() < probe:
        h_ratio = np.random.uniform(low=inf, high=sup)
        w_ratio = np.random.uniform(low=inf, high=sup)
        image, pts = scale(image, pts, h_ratio, w_ratio)
    return image, pts
'''
     
def extend_bbx_stage2(bbx, aspect_ratio):
    '''
    extend the bbx to make it 1:1 aspect ratio
    '''
    bbx = tf.reshape(bbx, [4, 1])
    wh = tf.matmul([[0.0, -1.0, 0.0, 1.0], [-aspect_ratio, 0.0, aspect_ratio, 0.0]], bbx)
    
    delta = 0.5 * tf.matmul([[-1.0, 1.0]], wh)
    
    delta = tf.tile(delta, [4, 1])
    print(delta.shape)
    
    delta = tf.multiply([[-1.0/aspect_ratio], [1.0], [-1.0/aspect_ratio], [1.0]], delta)
    
    delta = tf.maximum(0.0, delta)
    
    delta = tf.multiply([[-1.0], [-1.0], [1.0], [1.0]], delta)
    
    bbx = bbx + delta
    
    bbx = tf.reshape(bbx, [4])
    return bbx

def extend_bbx(bbx, aspect_ratio):
    bbx = extend_bbx_stage1(bbx)
    bbx = extend_bbx_stage2(bbx, aspect_ratio)
    return bbx


def get_extend_matrix(bbx):
    '''
    bbx: the extended bounding box
    returns a 3x3 matrix indicating the transform of the extending the image
    '''
    ymin = bbx[0]
    xmin = bbx[1]
    ymax = bbx[2]
    xmax = bbx[3]
    exmin = tf.maximum(0.0, 0.0-xmin)
    exmax = tf.maximum(0.0, xmax-1.0)
    eymin = tf.maximum(0.0, 0.0-ymin)
    eymax = tf.maximum(0.0, ymax-1.0)
     
    ex = exmin+exmax
    fx = ex+1.0
    ey = eymin+eymax
    fy = ey+1.0
          
    tmp = [[1.0/fx, 0.0, exmin/fx],
           [0.0, 1.0/fy,eymin/fy],
           [0.0,0.0,1.0]]
    tmp = tf.convert_to_tensor(tmp)
    return tmp


def extend_image(image, extend_matrix):
    """
    image: 3-D Tensor of shape (H, W, C), does not care about the number of C
    """
    exmin = extend_matrix[0,2] / (extend_matrix[0,0])
    exmax = 1.0 / (extend_matrix[0, 0]) - exmin - 1.0
    eymin = extend_matrix[1,2] / (extend_matrix[1,1])
    eymax = 1.0 / (extend_matrix[1, 1]) - eymin - 1.0
    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    exmin = tf.cast(width * exmin, tf.int32)
    exmax = tf.cast(width * exmax, tf.int32)
    eymin = tf.cast(height * eymin,tf.int32)
    eymax = tf.cast(height * eymax,tf.int32)
    image = tf.pad(image, paddings=((eymin, eymax), (exmin, exmax), (0, 0)), mode='CONSTANT', constant_values=0)
    return image


def get_crop_matrix(bbx):
    ymin = bbx[0]
    xmin = bbx[1]
    ymax = bbx[2]
    xmax = bbx[3]
    cwidth = xmax - xmin
    cheight = ymax - ymin
 
    tmp = [[1.0/(cwidth),0.0,-xmin/(cwidth)],
                    [0.0,1.0/(cheight),-ymin/(cheight)],
                    [0.0, 0.0, 1.0]]
    tmp = tf.convert_to_tensor(tmp)
    return tmp


def crop_image(image, crop_matrix):
    """
    image: 3-D Tensor of shape (H, W, C), does not care about the number of C
    """
    xmin = -crop_matrix[0, 2] / (crop_matrix[0, 0])
    xmax = 1.0/(crop_matrix[0,0]) + xmin
    ymin = -crop_matrix[1, 2] / (crop_matrix[1, 1])
    ymax = 1.0/(crop_matrix[1,1]) + ymin
    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    xmin = tf.cast(width * xmin, tf.int64)
    xmax = tf.cast(width * xmax, tf.int64)
    ymin = tf.cast(height * ymin, tf.int64)
    ymax = tf.cast(height * ymax, tf.int64)
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
    pts = tf.pad(pts, paddings=((0,0), (0,1)), mode='CONSTANT', constant_values=1.0)
    pts = tf.matmul(pts, tf.transpose(transform_matrix))
    pts = pts[:, :2]
    return pts


def train_preprocess(image, landmark, bbx):
    image, trans_matrix = infer_preprocess(image, bbx)
    landmark = transform_pts(landmark, trans_matrix)
    return image, landmark


def infer_preprocess(image, bbx):
    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    aspect_ratio = h / w
    
    bbx = extend_bbx(bbx, aspect_ratio)
    extend_matrix = get_extend_matrix(bbx)

    ymin = bbx[0]
    xmin = bbx[1]
    ymax = bbx[2]
    xmax = bbx[3]
    bbx_pts = tf.convert_to_tensor([[xmin, ymin],
                                    [xmax, ymax]])
    


    bbx_pts = transform_pts(bbx_pts, extend_matrix)
    xmin = bbx_pts[0, 0]
    ymin = bbx_pts[0, 1]
    xmax = bbx_pts[1, 0]
    ymax = bbx_pts[1, 1]

    bbx_crop = tf.convert_to_tensor([ymin, xmin, ymax, xmax])
    crop_matrix = get_crop_matrix(bbx_crop)

    image = extend_image(image, extend_matrix)
    image = crop_image(image, crop_matrix)
    trans_matrix = tf.matmul(crop_matrix, extend_matrix)

    return image, trans_matrix

def infer_postprocess(landmark, trans_matrix):
    inv_matrix = tf.linalg.inv(trans_matrix)
    landmark = transform_pts(landmark, inv_matrix)
    landmark = tf.clip_by_value(landmark, 0.0, 1.0)
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



