import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def projective_inverse_warp(img, depth, pose, intrinsics, ret_flows=False):
    """Inverse warp a source image to the target image plane based on projection.

    Args:
    img: the source image [batch, height_s, width_s, 3]
    depth: depth map of the target image [batch, height_t, width_t]
    pose: target to source camera transformation matrix [batch, 4, 4]
    intrinsics: camera intrinsics [batch, 3, 3]
    ret_flows: whether to return the displacements/flows as well
    Returns:
    Source image inverse warped to the target image plane [batch, height_t,
    width_t, 3]
    """
    batch, height, width, _ = img.get_shape().as_list()
    # Construct pixel grid coordinates.
    pixel_coords = meshgrid_abs(batch, height, width)

    # Convert pixel coordinates to the camera frame.
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics)

    # Construct a 4x4 intrinsic matrix.
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch, 1, 1])
    intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
    intrinsics = tf.concat([intrinsics, filler], axis=1)

    # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
    # pixel frame.
    proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)

    output_img = tf.contrib.resampler.resampler(img, src_pixel_coords)
    if ret_flows:
        return output_img, src_pixel_coords - cam_coords
    else:
        return output_img

def cam2pixel(cam_coords, proj):
    """Transforms coordinates in a camera frame to the pixel frame.

    Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]
    Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
    """
    batch, _, height, width = cam_coords.get_shape().as_list()
    cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
    unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
    xy_u = unnormalized_pixel_coords[:, 0:2, :]
    z_u = unnormalized_pixel_coords[:, 2:3, :]
    pixel_coords = xy_u / (z_u + 1e-10)
    pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])

    return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])


def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
    """Transforms coordinates in the pixel frame to the camera frame.

    Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates
    Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
    """
    batch, height, width = depth.get_shape().as_list()
    depth = tf.reshape(depth, [batch, 1, -1])
    pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
    cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords) * depth

    if is_homogeneous:
        ones = tf.ones([batch, 1, height*width])
        cam_coords = tf.concat([cam_coords, ones], axis=1)

    cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])

    return cam_coords


def meshgrid_abs(batch, height, width, is_homogeneous=True):
    """Construct a 2D meshgrid in the absolute coordinates.

    Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
    Returns:
    x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
    """
    xs = tf.linspace(0.0, tf.cast(width-1, tf.float32), width)
    ys = tf.linspace(0.0, tf.cast(height-1, tf.float32), height)
    xs, ys = tf.meshgrid(xs, ys)

    if is_homogeneous:
        ones = tf.ones_like(xs)
        coords = tf.stack([xs, ys, ones], axis=0)
    else:
        coords = tf.stack([xs, ys], axis=0)
    coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
    return coords

