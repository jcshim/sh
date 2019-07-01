from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os, time

from models.networks import Encoder, Decoder
import models.helpers as helpers
from models.ops_utils import conv2d
from geometry.projection import projective_inverse_warp

class Model(object):
    def __init__(self, FLAGS):
        self.config = FLAGS
        self.batch_size = self.config.batch_size
        self.data = self.config.data
        self.loss_weight = 10

        self.global_step = tf.train.get_or_create_global_step(graph=None) # Tensor
        self.learning_rate = self.config.learning_rate
        self.beta1 = self.config.beta1
        self.checkpoint_dir = self.config.checkpoint_dir

        self.continue_train = self.config.continue_train
        self.max_steps = self.config.max_steps
        self.summary_freq = self.config.summary_freq
        self.save_latest_freq = self.config.save_latest_freq


    def into_depth_and_rgb_block(self, raw_src_image, raw_src_depth, pose, reuse_weights=None):
        b, h, w, _ = raw_src_image.get_shape().as_list()
        z_size = 856
        z_geo_size = 600

        with tf.name_scope('preprocessing'):
            src_image = self.image2tensor(raw_src_image)
            if len(raw_src_depth.get_shape()) != 4:
                src_depth = tf.expand_dims(raw_src_depth, axis=3)
            else:
                src_depth = raw_src_depth
            # self.manual_check = pose

        with tf.name_scope('concat_rgbd'):
            #conv_depth = conv2d(raw_src_depth, 32, is_train=True, k_h=3, k_w=3, s=1)
            #conv_rgb = conv2d(src_image, 32*3, is_train=True, k_h=3, k_w=3, s=1)
            input_rgbd = tf.concat([src_image, src_depth], axis=3)

        with tf.name_scope('Encoder'):
            z_enc_out = Encoder(input_rgbd, num_outputs=z_size, reuse_weights=reuse_weights)
            _, z_h, z_w, _ = z_enc_out.get_shape().as_list()
            # print('encoder out', z_enc_out)

            # transform latent vector
            z_geo = tf.reshape(z_enc_out[:, :, :, :z_geo_size], [b, -1, 4])
            z_app = z_enc_out[:, :, :, z_geo_size:]
            # print('z geo', z_geo)
            # print('z app', z_app)

            z_geo_tf = tf.matmul(z_geo, pose)
            # print('z geo tf', z_geo_tf)
            # print('inv_RT', inv_RT)

            z_geo_tf = tf.reshape(z_geo_tf, [b, 1, 1, 600])  # TODO: solving z_h and z_w values
            z_tf = tf.concat([z_geo_tf, z_app], axis=3)

        with tf.name_scope('Depth'):
            if self.data == 'car':
                depth_bias = 2
                depth_scale = 1.0
                # self.depth_scale_vis = 125. / depth_scale
                # self.depth_bias_vis = depth_bias - depth_scale

            depth_dec_out = Decoder(z_geo_tf, 1, variable_scope='Depth_Decoder', reuse_weights=reuse_weights)
            depth_pred = depth_scale * tf.nn.tanh(depth_dec_out) + depth_bias

        with tf.name_scope('Pixel'):
            pixel_dec_out = Decoder(z_tf, 3, variable_scope='Pixel_Decoder', reuse_weights=reuse_weights)
            pixel_pred = tf.nn.tanh(pixel_dec_out)
            # print('pixel pred', pixel_pred)

        # with tf.name_scope('prediction'):
            # warped_pred = projective_inverse_warp(src_image, tf.squeeze(depth_pred), RT, intrinsic, ret_flows=False)
            # print('warped pred', warped_pred)


        # Collect output tensors
        pred = {}
        pred['out_depth'] = depth_pred
        pred['out_pixel'] = pixel_pred
        # pred['warped_image'] = warped_pred
        # pred['inverse_warping_image'] = tgt_img_tf
        # pred['tgt_image'] = fake_tgt

        return pred

    def build_train_graph(self, inputs):
        with tf.name_scope('input_data'):
            raw_src_image = inputs['B']
            raw_tgt_image = inputs['A']
            raw_src_depth = inputs['B_depth']
            raw_tgt_depth = inputs['A_depth']
            # RT = inputs['RT']
            intrinsic = inputs['intrinsic']
            RT, inv_RT = self.reshape_posematrix(inputs['RT'])

            # self.manual_check2 = inputs

        with tf.name_scope('inference'):
            with tf.name_scope('src2tgt'):
                predictions1 = self.into_depth_and_rgb_block(raw_src_image, raw_src_depth, RT)
                pred_tgt_image = predictions1['out_pixel']
                pred_tgt_depth = predictions1['out_depth']
            with tf.name_scope('tgt2src'):
                input_pred_tgt_image = self.tensor2image(pred_tgt_image)
                predictions2 = self.into_depth_and_rgb_block(input_pred_tgt_image, pred_tgt_depth, inv_RT, reuse_weights=tf.AUTO_REUSE)
                pred_src_image = predictions2['out_pixel']
                pred_src_depth = predictions2['out_depth']

        with tf.name_scope('loss'):
            if self.data =='car':
                # pixel value=1 if foreground otherwise pixel value=0
                fg_src_mask = tf.cast(raw_src_depth > 0, dtype=tf.float32)  # b x 256 x 256
                #fg_src_mask = tf.expand_dims(fg_src_mask, axis =-1) # b x 256 x256 x 1
            src_image = self.image2tensor(raw_src_image)
            src_depth = tf.expand_dims(raw_src_depth, axis=-1)
            # pred_src_depth = tf.squeeze(pred_src_depth)
            # pred_tgt_depth = tf.squeeze(pred_tgt_depth)

            # loss 1
            depth_loss = helpers.masked_L1_Loss(pred_src_depth, src_depth, mask=fg_src_mask)
            pixel_loss = helpers.masked_L1_Loss(pred_src_image, src_image, mask=fg_src_mask)

            # loss 2
            # create multiscale-pyramid
            photometric_loss1 = self.compute_photometric_loss(pred_tgt_depth, src_image, pred_tgt_image,
                                                                 intrinsic, RT, mask=None)
            photometric_loss2 = self.compute_photometric_loss(pred_src_depth, pred_tgt_image, src_image,
                                                                 intrinsic, inv_RT, mask=fg_src_mask)

            total_loss = depth_loss + pixel_loss + photometric_loss1 * 0.1 + photometric_loss2

        with tf.name_scope('train_op'):
            train_vars = [var for var in tf.trainable_variables()]
            optim = tf.train.AdamOptimizer(self.learning_rate, self.beta1)
            grads_and_vars = optim.compute_gradients(total_loss, var_list=train_vars)
            train_op = optim.apply_gradients(grads_and_vars)

        # Summaries
        tgt_img_sample = projective_inverse_warp(self.image2tensor(raw_src_image), raw_tgt_depth, RT, intrinsic, ret_flows=False)
        tgt_img_sample2 = projective_inverse_warp(self.image2tensor(raw_tgt_image), raw_src_depth, inv_RT, intrinsic, ret_flows=False)
        tf.summary.image('sample_image', tgt_img_sample)
        tf.summary.image('sample_image', tgt_img_sample2)

        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('depth_loss', depth_loss)
        tf.summary.scalar('pixel_loss', pixel_loss)
        tf.summary.scalar('photometric_loss1', photometric_loss1)
        tf.summary.scalar('photometric_loss2', photometric_loss2)

        tf.summary.image('raw_src_image', raw_src_image)
        tf.summary.image('raw_tgt_image', raw_tgt_image)
        tf.summary.image('raw_src_depth', tf.expand_dims(raw_src_depth, axis=-1))
        tf.summary.image('raw_tgt_depth', tf.expand_dims(raw_tgt_depth, axis=-1))
        tf.summary.image('pred_tgt_image', self.tensor2image(pred_tgt_image))
        tf.summary.image('pred_src_image', self.tensor2image(pred_src_image))
        tf.summary.image('pred_src_depth', pred_src_depth)
        tf.summary.image('pred_tgt_depth', pred_tgt_depth)


        return train_op

    def train(self, train_op):
        parameter_count = tf.reduce_sum(
            [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
        global_step = tf.Variable(0, name='global_step', trainable=False)
        incr_global_step = tf.assign(global_step, global_step + 1)
        saver = tf.train.Saver(
            [var for var in tf.model_variables()] + [global_step], max_to_keep=10)
        sv = tf.train.Supervisor(
            logdir=self.checkpoint_dir, save_summaries_secs=0, saver=None)

        with sv.managed_session() as sess:
            tf.logging.info('Trainable variables:')
            for var in tf.trainable_variables():
                tf.logging.info(var.name)

            tf.logging.info('parameter_count = %d' % sess.run(parameter_count))

            if self.continue_train:
                checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
                if checkpoint is not None:
                    tf.logging.info('Resume training from previous checkpoint')
                    saver.restore(sess, checkpoint)

            for step in range(1, self.max_steps):
                start_time = time.time()
                fetches = {
                    'train': train_op,
                    'global_step': global_step,
                    'incr_global_step': incr_global_step,
                }
                if step % self.summary_freq == 0:
                    fetches['summary'] = sv.summary_op

                # aaaa = sess.run(self.manual_check2)
                results = sess.run(fetches)
                gs = results['global_step']

                if step % self.summary_freq == 0:
                    sv.summary_writer.add_summary(results['summary'], gs)
                    tf.logging.info(
                        '[Step %.8d] time: %4.4f/it' % (gs, time.time() - start_time))

                if step % self.save_latest_freq == 0:
                    tf.logging.info(' [*] Saving checkpoint to %s...' % self.checkpoint_dir)
                    saver.save(sess, os.path.join(self.checkpoint_dir, 'model.latest'))


    def image2tensor(self, image):
        """Preprocess the image for CNN input.

        Args:
          image: the input image in uint8 [0, 255]
        Returns:
          A new image converted to float with range [-1, 1]
        """
        image = tf.image.convert_image_dtype(image, dtype=tf.float32) /255
        return image * 2 - 1

    def tensor2image(self, image):
        """Undo the preprocessing.

        Args:
          image: the input image in float with range [-1, 1]
        Returns:
          A new image converted to uint8 [0, 255]
        """
        image = (image + 1.) / 2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

    def reshape_posematrix(self, RT):
        tfvec = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor([0, 0, 0, 1], dtype=tf.float32), 0), 0)
        tfvec = tf.tile(tfvec, [self.batch_size, 1, 1])
        pose4x4 = tf.concat([RT, tfvec], axis =1)  # tgt to src pose
        inv_pose4x4 = tf.matrix_inverse(pose4x4)

        return pose4x4, inv_pose4x4

    def compute_photometric_loss(self, pred_depth, rgb_src, rgb_tgt, intrinsic, RT, mask=None):
        pred_array = helpers.multiscale(pred_depth)
        rgb_src_array = helpers.multiscale(rgb_src)
        rgb_tgt_array = helpers.multiscale(rgb_tgt)
        if mask is not None:
            mask_array = helpers.multiscale(mask)

        photometric_loss =0
        num_scales = len(pred_array)

        for scale in range(len(pred_array)):
            pred_ = pred_array[scale]
            rgb_src_ = rgb_src_array[scale]
            rgb_tgt_ = rgb_tgt_array[scale]
            mask_ = None
            if mask is not None:
                mask_ = mask_array[scale]

            # compute the corresponding intrinsic parameters
            b, h_, w_, c = pred_.get_shape().as_list()
            ratio_h, ratio_w = h_/256, w_/256
            intrinsic_ = helpers.scale_intrinsics(intrinsic, ratio_h, ratio_w)

            # inverse warp from a nearby frame to the current frame
            pred_ = tf.squeeze(pred_)
            warped_ = projective_inverse_warp(rgb_src_, pred_, RT, intrinsic_, ret_flows=False)

            photometric_loss += helpers.photometric_Loss(warped_, rgb_tgt_, mask_) * (2 ** (scale - num_scales))

        return photometric_loss