# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------

from numba import jit
import numpy as np

EPS = 1e-3

@jit(nopython=True)
def compute_piou(x1,y1,a1,b1,c1, x2,y2,a2,b2,c2):
    
    B1 = 1/4.*( (a1+a2)*(y1-y2)**2. + (b1+b2)*(x1-x2)**2. )/( (a1+a2)*(b1+b2) - (c1+c2)**2. + EPS )
    sqrt = (a1*b1-c1**2)*(a2*b2-c2**2)
    B2 = ( (a1+a2)*(b1+b2) - (c1+c2)**2. )/( 4.*np.sqrt(sqrt) + EPS )
    B2 = 1/2.*np.log(B2 + EPS)
    
    Db = B1 + B2
    
    return 1. - np.sqrt(1. - np.exp(-Db))

@jit(nopython=True)
def get_piou_values(array, angles):
    # xmin, ymin, xmax, ymax
    xmin = array[0]; ymin = array[1]
    xmax = array[2]; ymax = array[3]
    
    # get ProbIoU values without rotation
    x = (xmin + xmax)/2.
    y = (ymin + ymax)/2.
    a = np.power((xmax - xmin), 2.)/12.
    b = np.power((ymax - ymin), 2.)/12.
    
    # convert values to rotations
    a = a*np.power(np.cos(angles), 2.) + b*np.power(np.sin(angles), 2.)
    b = a*np.power(np.sin(angles), 2.) + b*np.power(np.cos(angles), 2.)
    c = a*np.cos(angles)*np.sin(angles) - b*np.sin(angles)*np.cos(angles)
    return x, y, a, b, c

@jit(nopython=True)
def compute_overlap(boxes, query_boxes):
    """
    Args
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    
    for k in range(K):
        
        for n in range(N):
            
            angle = boxes[n][...,-1]
            
            overlaps[n, k] = compute_piou(
                *get_piou_values(query_boxes[k], angle),
                *get_piou_values(boxes[n][...,:4], angle)
            )
            
    return overlaps
