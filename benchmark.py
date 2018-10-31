#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg') # enable this to plot correctly on server
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline
from face_landmark import metric
def plot_cdf(x):
    num_bins = x.shape[0]
    counts, bin_edges = np.histogram (x, bins=num_bins, normed=True)
    cdf = np.cumsum (counts)
    plt.xlim(0, np.max(x))
    plt.ylim(0, 1)

    x_axis = np.insert(bin_edges[1:], 0, 0)
    y_axis = np.insert(cdf/cdf[-1], 0, 0)

    x_axis_finegrain = np.linspace(x_axis.min(), x_axis.max(), int(90000 / num_bins) + num_bins)
    y_axis_smooth = spline(x_axis ,y_axis,x_axis_finegrain)
    plt.plot (x_axis_finegrain, y_axis_smooth)
    plt.savefig('tmp.png')
    return

pts = np.load('/barn2/yuan/westworld/pts_dump3.npy')
pts_real = np.load('/barn2/yuan/westworld/pts_real_dump3.npy')
d = np.load('/barn2/yuan/westworld/d_dump3.npy')


nme = metric.nme_batch(pts, pts_real, d)
print(nme)

pts = np.load('/barn2/yuan/westworld/pts_dump2.npy')
pts_real = np.load('/barn2/yuan/westworld/pts_real_dump2.npy')
d = np.load('/barn2/yuan/westworld/d_dump2.npy')


nme = metric.nme_batch(pts, pts_real, d)
print(nme)