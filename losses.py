"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# import keras
import math
from tensorflow import keras
import tensorflow as tf
import numpy as np

from utils.anchors import anchors_for_shape
from layers import RegressBoxes

def focal(alpha=0.25, gamma=1.5):
    """
    Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """

    def _focal(y_true, y_pred):
        """
        Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels = y_true[:, :, :-1]
        # -1 for ignore, 0 for background, 1 for object
        anchor_state = y_true[:, :, -1]
        classification = y_pred

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.not_equal(anchor_state, -1))
        labels = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = tf.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
        focal_weight = tf.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma
        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return keras.backend.sum(cls_loss) / normalizer
        
        #loss = tf.math.divide_no_nan(keras.backend.sum(cls_loss), normalizer)
        #return tf.where(tf.math.is_nan(loss), 0., loss)

    return _focal


def smooth_l1(sigma=3.0):
    """
    Create a smooth L1 loss functor.
    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.
    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.
        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).
        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1


def smooth_l1_quad(sigma=3.0):
    """
    Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression = tf.concat([regression[..., :4], tf.sigmoid(regression[..., 4:9])], axis=-1)
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        box_regression_loss = tf.where(
            keras.backend.less(regression_diff[..., :4], 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff[..., :4], 2),
            regression_diff[..., :4] - 0.5 / sigma_squared
        )

        alpha_regression_loss = tf.where(
            keras.backend.less(regression_diff[..., 4:8], 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff[..., 4:8], 2),
            regression_diff[..., 4:8] - 0.5 / sigma_squared
        )

        ratio_regression_loss = tf.where(
            keras.backend.less(regression_diff[..., 8], 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff[..., 8], 2),
            regression_diff[..., 8] - 0.5 / sigma_squared
        )
        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())

        box_regression_loss = tf.reduce_sum(box_regression_loss) / normalizer
        alpha_regression_loss = tf.reduce_sum(alpha_regression_loss) / normalizer
        ratio_regression_loss = tf.reduce_sum(ratio_regression_loss) / normalizer

        return box_regression_loss + alpha_regression_loss + 16 * ratio_regression_loss

    return _smooth_l1

''' Probabilistic IoU '''

EPS = 1e-3

def helinger_dist(x1,y1,a1,b1,c1, x2,y2,a2,b2,c2, freezed=False):
    
    B1 = 1/4.*( (a1+a2)*(y1-y2)**2. + (b1+b2)*(x1-x2)**2. )/( (a1+a2)*(b1+b2) - (c1+c2)**2. + EPS )
    if freezed:
        B2 = 0.
    else:
        sqrt = (a1*b1-c1**2)*(a2*b2-c2**2)
        sqrt = tf.where(sqrt<0, EPS, sqrt)
        B2 = ( (a1+a2)*(b1+b2) - (c1+c2)**2. )/( 4.*tf.math.sqrt(sqrt) + EPS )
        B2 = tf.where(B2<0, EPS, B2)
        B2 = 1/2.*tf.math.log(B2 + EPS)
    
    Bd = B1 + B2
        
    Db = tf.clip_by_value(Bd, EPS, 100.)
    
    return tf.math.sqrt(1 - tf.math.exp(-Db) + EPS)

def get_piou_values(points):
    # xmin, ymin, xmax, ymax
    xmin = points[:,0]; ymin = points[:,1]
    xmax = points[:,2]; ymax = points[:,3]
    angles = points[:,-1]
    
    # get ProbIoU values without rotation
    x = (xmin + xmax)/2.
    y = (ymin + ymax)/2.
    a = tf.math.pow((xmax - xmin), 2.)/12.
    b = tf.math.pow((ymax - ymin), 2.)/12.
    
    # convert values to rotations
    a = a*tf.math.pow(tf.math.cos(angles), 2.) + b*tf.math.pow(tf.math.sin(angles), 2.)
    b = a*tf.math.pow(tf.math.sin(angles), 2.) + b*tf.math.pow(tf.math.cos(angles), 2.)
    c = a*tf.math.cos(angles)*tf.math.sin(angles) - b*tf.math.sin(angles)*tf.math.cos(angles)
    return x, y, a, b, c

def calc_piou(mode, regression_target, regression, freezed=False):
    
    l1 = helinger_dist(
                *get_piou_values(regression_target),
                *get_piou_values(regression),
                freezed=freezed
            )
    if mode=='piou_l1':
        return l1
    
    l2 = tf.math.pow(l1, 2.)
    if mode=='piou_l2':
        return l2
    
    l3 = - tf.math.log(1. - l2 + EPS)
    if mode=='piou_l3':
        return l3

def iou_loss(mode, phi, weight, anchor_parameters=None, freeze_iterations=0):
    
    assert phi in range(7)
    image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
    input_size = float(image_sizes[phi])
    it = 0
    
    def _iou(y_true, y_pred):
        nonlocal it
        
        # separate target and state
        regression = y_pred[..., :5]
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]
        
        # convert to boxes values: xmin, ymin, xmax, ymax
        anchors = anchors_for_shape((input_size, input_size), anchor_params=anchor_parameters)
        anchors_input = np.expand_dims(anchors, axis=0)
        regression = RegressBoxes(name='boxes')([anchors_input, regression])
        regression_target = RegressBoxes(name='boxes')([anchors_input, regression_target])

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)
        
        loss = calc_piou(mode, regression_target, regression, freezed=freeze_iterations>it)
        it += 1
        
        return tf.cast(weight, 'float32') * loss

    return _iou