import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def instance_norm(input):
    """ instance normalization """
    with tf.variable_scope('instance_norm'):
        num_out = input.get_shape()[-1]
        scale = tf.get_variable(
            'scale', [num_out],
            initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable(
            'offset', [num_out],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
        mean, var = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-6
        inv = tf.rsqrt(var + epsilon)
        return scale * (input - mean) * inv + offset


def batchnorm_activation(input, is_train, norm='batch', activation_fn=None, name="bn_act"):
    with tf.variable_scope(name):
        _ = input
        if activation_fn is not None:
            _ = activation_fn(_)
        if norm is not None and norm is not False:
            if norm == 'batch':
                # batch norm
                _ = tf.contrib.layers.batch_norm(
                    _, center=True, scale=True, decay=0.999,
                    is_training=is_train, updates_collections=None
                )
            elif norm == 'instance':
                _ = instance_norm(_)
            elif norm == 'None':
                _ = _
    return _


def conv2d(input, output_shape, is_train, k_h=4, k_w=4, s=2, name="conv2d", activation_fn=lrelu, norm='batch'):
    with tf.variable_scope(name):
        _ = slim.conv2d(input, output_shape, [k_h, k_w], stride=s, activation_fn=None)
        _ = batchnorm_activation(_, is_train, norm=norm, activation_fn=activation_fn)
    return _


def bilinear_deconv2d(input, output_shape, is_train, upsample_scale =2, k_h=3, k_w=3, name="bilinear_deconv2d", activation_fn=lrelu, norm='batch'):
    with tf.variable_scope(name):
        new_h = int(input.get_shape()[1]) * upsample_scale
        new_w = int(input.get_shape()[2]) * upsample_scale
        _ = tf.image.resize_bilinear(input, [new_h, new_w])
        _ = slim.conv2d(_, output_shape, [k_h, k_w], stride =1, activation_fn=None)
        _ = batchnorm_activation(_, is_train, norm=norm, activation_fn=activation_fn)
    return _


def deconv2d(input, output_shape, is_train, k =4, s=2, name="deconv2d", stddev=0.01, activation_fn=lrelu, norm='batch'):
    with tf.variable_scope(name):
        # _ = layers.conv2d_transpose(input, num_outputs=output_shape, kernel_size=[k, k], stride=[s, s], padding='SAME',
        #     weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
        #     biases_initializer=tf.zeros_initializer(), activation_fn=None
        _ = slim.conv2d_transpose(input, output_shape, [k, k], stride=s, padding='SAME',
                                  weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                  biases_initializer=tf.zeros_initializer(), activation_fn=None)

        _ = batchnorm_activation(_, is_train, norm=norm, activation_fn=activation_fn)
    return _


def residual_conv(input, num_filters, filter_size, stride, reuse=False,
                  pad='SAME', dtype=tf.float32, bias=False, name='res_conv'):
    with tf.variable_scope(name):
        stride_shape = [1, stride, stride, 1]
        filter_shape = [filter_size, filter_size, input.get_shape()[3], num_filters]
        w = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
        p = (filter_size - 1) // 2
        x = tf.pad(input, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
        conv = tf.nn.conv2d(x, w, stride_shape, padding='VALID')
    return conv


def residual(input, num_filters, name, is_train, reuse=False, pad='REFLECT'):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('res1', reuse=reuse):
            out = residual_conv(input, num_filters, 3, 1, reuse, pad, name=name)
            out = tf.contrib.layers.norm(
                out, center=True, scale=True, decay=0.9,
                is_training=is_train, updates_collections=None
            )
            out = tf.nn.relu(out)

        with tf.variable_scope('res2', reuse=reuse):
            out = residual_conv(out, num_filters, 3, 1, reuse, pad, name=name)
            out = tf.contrib.layers.norm(
                out, center=True, scale=True, decay=0.9,
                is_training=is_train, updates_collections=None
            )

        return tf.nn.relu(input + out)
