import numpy as np
import metric


def nme_batch_test():
    np.random.seed(100)
    x = np.random.normal(loc=10.0, scale=1.0, size=(255, 68, 2)).astype(np.float32)
    y = np.random.normal(loc=-5.0, scale=1.0, size=(255, 68, 2)).astype(np.float32)
    d = np.sqrt(32**2+30**2)
    res = metric.nme_batch(x, y, d)
    if res < 0.2507503005342272 and res > 0.2507503005342270:
        return True
    return False

def nme_test():
    np.random.seed(100)
    x = np.random.normal(loc=10.0, scale=1.0, size=(68, 2)).astype(np.float32)
    y = np.random.normal(loc=10.0, scale=1.0, size=(68, 2)).astype(np.float32)
    d = np.sqrt(32*30)
    res = metric.nme(x, y, d)
    if res < 0.5357806838856682 and res > 0.5357806838856680:
        return True
    return False



if __name__ == '__main__':
    try:
        assert nme_batch_test()
        print(metric.__file__, 'PASSED test')
    except:
        print(metric.__file__, 'FAILED test')

    try:
        assert nme_test()
        print(metric.__file__, 'PASSED test')
    except:
        print(metric.__file__, 'FAILED test')



