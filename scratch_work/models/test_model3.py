from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os, time

from models.networks import Encoder, Decoder
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

    def infer_tgt_views(self, raw_src_image, RT, intrinsic):
        b, h, w, _ = raw_src_image.get_shape().as_list()
        z_size = 856

        with tf.name_scope('preprocessing'):
            src_image = self.image2tensor(raw_src_image)
            self.manual_check = RT
            RT, inv_RT = self.reshape_posematrix(RT)



        with tf.name_scope('Encoder'):
            z_enc_out = Encoder(src_image, num_outputs=z_size)
            _, z_h, z_w, _ = z_enc_out.get_shape().as_list()
            # print('encoder out', z_enc_out)

            # transform latent vector
            z_geo = tf.reshape(z_enc_out[:, :, :, :600], [b, -1, 4])
            z_app = z_enc_out[:, :, :, 600:]
            # print('z geo', z_geo)
            # print('z app', z_app)

            z_geo_tf = tf.matmul(z_geo, inv_RT)
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

            depth_dec_out = Decoder(z_geo_tf, 1, variable_scope='Depth_Decoder')
            depth_pred = depth_scale * tf.nn.tanh(depth_dec_out) + depth_bias

        with tf.name_scope('Mask'):
            mask_dec_out = Decoder (z_geo_tf, 1,  variable_scope='Mask_Decoder')
            mask_pred = tf.nn.sigmoid(mask_dec_out)
            # print('mask pred', mask_pred)

        with tf.name_scope('Pixel'):
            pixel_dec_out = Decoder(z_tf, 3, variable_scope='Pixel_Decoder')
            pixel_pred = tf.nn.tanh(pixel_dec_out)
            # print('pixel pred', pixel_pred)

        with tf.name_scope('prediction'):
            warped_pred = projective_inverse_warp(src_image, tf.squeeze(depth_pred), RT, intrinsic, ret_flows=False)
            # print('warped pred', warped_pred)

            fake_tgt = tf.multiply(pixel_pred, mask_pred) + tf.multiply(warped_pred, 1-mask_pred)

        # Collect output tensors
        pred = {}
        pred['out_depth'] = depth_pred
        pred['out_mask'] = mask_pred
        pred['out_pixel'] = pixel_pred
        pred['warped_image'] = warped_pred
        pred['tgt_image'] = fake_tgt

        return pred

    def build_train_graph(self, inputs):
        with tf.name_scope('input_data'):
            raw_src_image = inputs['B']
            raw_tgt_image = inputs['A']
            raw_src_depth = inputs['B_depth']
            raw_tgt_depth = inputs['A_depth']
            RT = inputs['RT']
            intrinsic = inputs['intrinsic']

            self.manual_check2 = inputs

        with tf.name_scope('inference'):
            predictions = self.infer_tgt_views(raw_src_image, RT, intrinsic)
            out_pixel = predictions['out_pixel']
            warped_pred = predictions['warped_image']
            tgt_image_pred = predictions['tgt_image']

        with tf.name_scope('loss'):
            tgt_image = self.image2tensor(raw_tgt_image)
            depth_loss = tf.reduce_mean(tf.abs(tgt_image - warped_pred)) * self.loss_weight
            pixel_loss = tf.reduce_mean(tf.abs(tgt_image - out_pixel)) * self.loss_weight
            mask_loss = tf.reduce_mean(tf.abs(tgt_image - tgt_image_pred)) * self.loss_weight

            total_loss = depth_loss + pixel_loss + mask_loss

        with tf.name_scope('train_op'):
            train_vars = [var for var in tf.trainable_variables()]
            optim = tf.train.AdamOptimizer(self.learning_rate, self.beta1)
            grads_and_vars = optim.compute_gradients(total_loss, var_list=train_vars)
            train_op = optim.apply_gradients(grads_and_vars)

        # Summaries
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.image('raw_src_image', raw_src_image)
        tf.summary.image('raw_tgt_image', raw_tgt_image)
        tf.summary.image('pred_tgt_image', self.tensor2image(tgt_image_pred))
        tf.summary.image('warped_image', self.tensor2image(warped_pred))
        tf.summary.image('pixel_out', self.tensor2image(out_pixel))



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

                aaaa = sess.run(self.manual_check2)
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
          image: the input image in either float [0, 1] or uint8 [0, 255]
        Returns:
          A new image converted to float with range [-1, 1]
        """
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
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
