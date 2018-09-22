import numpy as np

def nme_batch(x: np.ndarray, y: np.ndarray, d):
    '''
    x, y are (N, 68, 2) alignment coordinates,
    d is the normalized dimension of boundbing box
    it is the square root w_bbox * h_bbox of the bounding box (area of the ground truth bounding box)
    '''
    n = x.shape[0]
    assert n == y.shape[0]
    assert x.shape[1] == 68
    assert x.shape[2] == 2
    assert y.shape[1] == 68
    assert y.shape[2] == 2
    norm_sum = np.linalg.norm(x - y)
    res = norm_sum / (n * float(d))
    return res

def nme(x: np.ndarray, y: np.ndarray, d):
    '''
    x, y are (68, 2) alignment coordinates
    '''
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[0] == 68
    assert y.shape[0] == 68
    assert x.shape[1] == 2
    assert y.shape[1] == 2
    norm_sum = np.mean(np.linalg.norm(x - y, axis=1))
    #norm_sum = np.mean(np.linalg.norm(x - y))
    res = norm_sum / d
    return res

def nme_v2(x: np.ndarray, y: np.ndarray, d):
    '''
    x, y are (68, 2) alignment coordinates
    '''
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[0] == 68
    assert y.shape[0] == 68
    assert x.shape[1] == 2
    assert y.shape[1] == 2
    #norm_sum = np.mean(np.linalg.norm(x - y, axis=1))
    norm_sum = np.mean(np.linalg.norm(x - y))
    res = norm_sum / d
    return res


