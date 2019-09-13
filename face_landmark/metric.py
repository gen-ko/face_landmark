import numpy as np

def nme_batch_v1(x: np.ndarray, y: np.ndarray, d):
    """
    x, y are (N, 68, 2) alignment coordinates,
    d (N, )
    d is the normalized dimension of boundbing box
    or, as bounding boxes are ambiguous, use the interocular distance as d
    it is the square root w_bbox * h_bbox of the bounding box (area of the ground truth bounding box)
    """
    n = x.shape[0]
    assert n == y.shape[0]
    try:
        assert x.shape[1] == 68
    except AssertionError:
        print('assert x.shape[1] == 68 error', x.shape)
        exit(-1)
    try:
        assert x.shape[2] == 2
    except AssertionError:
        print('assert x.shape[2] == 2 error', x.shape)
        exit(-1)
    try:
        assert y.shape[1] == 68
    except AssertionError:
        print('assert y.shape[1] == 68', y.shape)
        exit(-1)
    try:
        assert y.shape[2] == 2
    except AssertionError:
        print('assert y.shape[2] == 2', y.shape)
        exit(-1)
    # (N, 68)
    norm_sum = np.linalg.norm(x - y, axis=2)
    # (N)
    nme_sample = np.sum(norm_sum, axis=1) / (norm_sum.shape[1] * d)
    res = np.mean(nme_sample, axis=0)
    return res

def nme(x: np.ndarray, y: np.ndarray, d=None):
    """
    x, y are (68, 2) alignment coordinates
    x is assumed to be an array of groundtruth landmark points
    y is assumed to be an array of detection landmark points
    z
    """
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[0] == 68
    assert y.shape[0] == 68
    assert x.shape[1] == 2
    assert y.shape[1] == 2
    if d is None:
        w = x[:, 0].max() - x[:, 0].min()
        h = x[:, 1].max() - x[:, 1].min()
        d = np.sqrt(w * h)
    norm_sum = np.mean(np.linalg.norm(x - y, axis=1))
    #norm_sum = np.mean(np.linalg.norm(x - y))
    res = norm_sum / d
    return res



def nme_batch(x: np.ndarray, y: np.ndarray, d=None) -> np.ndarray:
    """
    x: gt y:
    x, y are (N, 68, 2) alignment coordinates,
    d (N, )
    d is the normalized dimension of boundbing box
    or, as bounding boxes are ambiguous, use the interocular distance as d
    it is the square root w_bbox * h_bbox of the bounding box (area of the ground truth bounding box)
    """
    n = x.shape[0]
    assert n == y.shape[0]
    try:
        assert x.shape[1] == 68
    except AssertionError:
        print('assert x.shape[1] == 68 error', x.shape)
        exit(-1)
    try:
        assert x.shape[2] == 2
    except AssertionError:
        print('assert x.shape[2] == 2 error', x.shape)
        exit(-1)
    try:
        assert y.shape[1] == 68
    except AssertionError:
        print('assert y.shape[1] == 68', y.shape)
        exit(-1)
    try:
        assert y.shape[2] == 2
    except AssertionError:
        print('assert y.shape[2] == 2', y.shape)
        exit(-1)
    if d is None:
        w = x[:,:,0].max(axis=1) - x[:,:,0].min(axis=1)
        h = x[:,:,1].max(axis=1) - x[:,:,1].min(axis=1)
        d = np.sqrt(w * h)
    
    # (N, 68)
    norm_sum = np.linalg.norm(x - y, axis=2) 
    # (N)
    nme_sample = np.sum(norm_sum, axis=1) / (norm_sum.shape[1] * d)
    #res = np.mean(nme_sample)
    return nme_sample 