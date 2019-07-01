import tensorflow as tf
import numpy as np

def multiscale(img):
    k = [1,2,2,1]
    s = [1,2,2,1]
    img1 = tf.nn.avg_pool(img, ksize=k, strides=s, padding='VALID')
    img2 = tf.nn.avg_pool(img1, ksize=k, strides=s, padding='VALID')
    img3 = tf.nn.avg_pool(img2, ksize=k, strides=s, padding='VALID')
    img4 = tf.nn.avg_pool(img3, ksize=k, strides=s, padding='VALID')
    img5 = tf.nn.avg_pool(img4, ksize=k, strides=s, padding='VALID')

    return [img1, img2, img3, img4, img5]

def scale_intrinsics(intrinsic, ratio_h, ratio_w):

    # return a new set of corresponding intrinsic parameters for the scaled image
    b, h, w = intrinsic.get_shape().as_list()
    fu = intrinsic[0,0,0]
    fv = intrinsic[0, 1, 1]
    cv = intrinsic[0, 0, 2]
    cu = intrinsic[0, 1, 2]

    fu = ratio_w * fu
    fv = ratio_h * fv
    cu = ratio_w * cu
    cv = ratio_h * cv
    new_intrinsics = tf.tile([fu, 0, cu, 0, fv, cv, 0, 0, 1], [b])
    new_intrinsics = tf.cast(tf.reshape(new_intrinsics, shape=[b,3,3]), dtype=tf.float32)

    return new_intrinsics

def masked_L1_Loss(pred, target, mask=None, weight=None):
    '''

    :param pred: A tensor that is predicted from the network
    :param target: A image tensor that is converted in a range between [-1, 1]
    :param weight:
    :return:
    '''
    assert pred.get_shape().as_list() == target.get_shape().as_list(), 'Inconsistent dimension'

    if mask is None:
        valid_mask = tf.cast(target < 1, dtype=tf.float32)
    else:
        valid_mask = mask

    diff = pred - target
    diff = tf.multiply(diff, valid_mask)
    loss = tf.reduce_mean(tf.abs(diff))

    return loss


def masked_L2_Loss(pred, target, mask=None, weight=None):

    assert pred.get_shape().as_list() == target.get_shape().as_list(), 'Inconsistent dimension'

    if mask is None:
        valid_mask = tf.cast(target < 1, dtype=tf.float32)
    else:
        valid_mask = mask

    diff = pred - target
    diff = tf.multiply(diff, valid_mask)
    loss = tf.reduce_mean(diff ** 2)

    return loss

def photometric_Loss(recon, target, mask=None, weight=None):
    assert len(recon.get_shape().as_list()) == 4, "expected recon dimension to be 4, but instead got {}.".format(recon.get_shape().as_list())
    assert len(target.get_shape().as_list()) == 4, "expected target dimension to be 4, but instead got {}.".format(target.get_shape().as_list())

    if mask is None:
        valid_mask = tf.cast(target < 1, dtype=tf.float32)
    else:
        valid_mask = mask

    diff = tf.abs(recon - target)
    diff = tf.multiply(diff, valid_mask)
    diff = tf.reduce_sum(diff, axis=3)  # sum along the color channel
    loss = tf.reduce_mean(diff)

    return loss
