from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from models.ops_utils import conv2d
from models.ops_utils import deconv2d, bilinear_deconv2d

from utils.print_utils import yellow, green, red, magenta, cyan


flags = tf.app.flags
flags.DEFINE_integer('num_layers', 7, 'Number of convolution layers.')
flags.DEFINE_string('batch_norm_type', 'batch', 'Type of the batch normalization activation. [batch | instance | None]')
FLAGS = flags.FLAGS

def Encoder(inputs, num_outputs=856, ngf=64, is_train=True, variable_scope='Encoder', reuse_weights=False):
    '''
        Encoder network definition
    :param inputs: stack of input images [batch, height, width, 3]
    :param num_outputs: number of output channels
    :param ngf: number of features for the first conv layer
    :param variable_scope: variable scope
    :param reuse_weights: whether to reuse weights (for weight sharing)
    :return:

    '''
    b,h,w,c = inputs.get_shape().as_list()

    with tf.variable_scope(variable_scope, reuse=reuse_weights):
        out = conv2d(inputs, 32, is_train, k_h=4, k_w=4, norm=FLAGS.batch_norm_type, name='conv1')
        out = conv2d(out, 64, is_train, k_h=4, k_w=4, norm=FLAGS.batch_norm_type, name='conv2')
        out = conv2d(out, 128, is_train, k_h=4, k_w=4, norm=FLAGS.batch_norm_type, name='conv3')
        out = conv2d(out, 256, is_train, k_h=4, k_w=4, norm=FLAGS.batch_norm_type, name='conv4')
        out = conv2d(out, 256, is_train, k_h=4, k_w=4, norm=FLAGS.batch_norm_type, name='conv5')
        out = conv2d(out, 256, is_train, k_h=4, k_w=4, norm=FLAGS.batch_norm_type, name='conv6')
        out = conv2d(out, 256, is_train, k_h=4, k_w=4, norm=FLAGS.batch_norm_type, name='conv7')
        out = conv2d(out, 256, is_train, k_h=4, k_w=4, norm=FLAGS.batch_norm_type, name='conv8')

        # Conv
        # enc_output = []
        # for i in range(FLAGS.num_layers):
        #     layer = conv2d(inputs, min(256, ngf*(i+1)), is_train, k_h=4, k_w=4, norm=FLAGS.batch_norm_type, name='conv{}'.format(i + 1))
        #     enc_output.append(layer)

        enc_output = tf.contrib.layers.fully_connected(out, num_outputs)

        return enc_output

def Decoder(inputs, num_outputs, is_train=True, variable_scope='Decoder', reuse_weights=False):
    with tf.variable_scope(variable_scope, reuse=reuse_weights):
        out = tf.contrib.layers.fully_connected(inputs, 256, activation_fn=tf.nn.leaky_relu)
        out = deconv2d(out, 256, is_train, name='deconv1')
        out = deconv2d(out, 256, is_train, name='deconv2')
        out = bilinear_deconv2d(out, 256, is_train, name='deconv3')
        out = bilinear_deconv2d(out, 256, is_train, name='deconv4')
        out = bilinear_deconv2d(out, 256, is_train, name='deconv5')
        out = bilinear_deconv2d(out, 128, is_train, name='deconv6')
        out = bilinear_deconv2d(out, 64, is_train, name='deconv7')
        out = bilinear_deconv2d(out, 32, is_train, name='deconv8')
        out = slim.conv2d(out, num_outputs, [3, 3], stride =1, activation_fn=None, scope='pred')

        return out


