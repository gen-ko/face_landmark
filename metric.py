import numpy as np

def nme_batch(x: np.ndarray, y: np.ndarray, d):
    '''
    x, y are (N, 68, 2) alignment coordinates,
    d (N, )
    d is the normalized dimension of boundbing box
    or, as bounding boxes are ambiguous, use the interocular distance as d
    it is the square root w_bbox * h_bbox of the bounding box (area of the ground truth bounding box)
    '''
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


