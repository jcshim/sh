from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os, time

from utils.print_utils import yellow, green, red, magenta, cyan
from models.networks import Encoder, Decoder
from geometry.projection import projective_inverse_warp

class Model(object):

    def __init__(self, flag,  debug_information=False, is_train=True):
        self.debug = debug_information

        self.config = flag
        self.batch_size = self.config.batch_size
        self.data = self.config.data
        self.loss_weight = 10

        self.global_step = tf.train.get_or_create_global_step(graph=None) # Tensor
        self.learning_rate = self.config.learning_rate
        self.beta1 = self.config.beta1
        self.checkpoint_dir = self.config.checkpoint_dir
        self.continue_train = self.config.continue_train

        # create placeholders for the input
        self.src_image = tf.placeholder(name='src_image', dtype=tf.float32, shape=[self.batch_size, 256, 256, 3])
        self.tgt_image = tf.placeholder(name='tgt_image', dtype=tf.float32, shape=[self.batch_size, 256, 256, 3])
        self.src_depth = tf.placeholder(name='src_depth', dtype=tf.float32, shape=[self.batch_size, 256, 256])
        self.tgt_depth = tf.placeholder(name='tgt_depth', dtype=tf.float32, shape=[self.batch_size, 256, 256])
        self.RT = tf.placeholder(name='RT', dtype=tf.float32, shape=[self.batch_size, 4, 4])
        self.inv_RT = tf.placeholder(name='inv_RT', dtype=tf.float32, shape=[self.batch_size, 4, 4])
        self.intrinsic = tf.placeholder(name='intrinsic', dtype=tf.float32, shape=[self.batch_size, 3,3])

        self.step = tf.placeholder(name='step', dtype=tf.int32, shape=[])
        self.is_train = tf.placeholder(name='is_train', dtype=tf.bool, shape=[])
        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

    def build_train_graph(self, is_train=True):
        z_size = 856

        with tf.name_scope('Encoder'):
            z_enc_out = Encoder(self.src_image, num_outputs=z_size)
            _, z_h, z_w, _ = z_enc_out.get_shape().as_list()
            print('encoder out', z_enc_out)


            # transform latent vector
            z_geo = tf.reshape(z_enc_out[:, :, :, :600], [self.batch_size, -1, 4])
            z_app = z_enc_out[:, :, :, 600:]
            print('z geo', z_geo)
            print('z app', z_app)

            z_geo_tf = tf.matmul(z_geo, self.inv_RT)
            print('z geo tf', z_geo_tf)
            print('inv_RT', self.inv_RT)

            z_geo_tf = tf.reshape(z_geo_tf, [self.batch_size, 1,1, 600]) #TODO: solving z_h and z_w values
            z_tf = tf.concat([z_geo_tf, z_app], axis=3)
            print('z tf', z_tf)

        with tf.name_scope('Depth'):
            if self.data == 'car':
                depth_bias = 2
                depth_scale = 1.0
                self.depth_scale_vis = 125. / depth_scale
                self.depth_bias_vis = depth_bias - depth_scale

            depth_dec_out = Decoder(z_geo_tf, 1, variable_scope='Depth_Decoder')
            depth_pred = depth_scale * tf.nn.tanh(depth_dec_out) + depth_bias


        with tf.name_scope('Mask'):
            mask_dec_out = Decoder (z_geo_tf, 1,  variable_scope='Mask_Decoder')
            mask_pred = tf.nn.sigmoid(mask_dec_out)
            print('mask pred', mask_pred)

        with tf.name_scope('Pixel'):
            pixel_dec_out = Decoder(z_tf, 3, variable_scope='Pixel_Decoder')
            pixel_pred = tf.nn.tanh(pixel_dec_out)
            print('pixel pred', pixel_pred)

        with tf.name_scope('prediction'):
            warped_pred = projective_inverse_warp(self.src_image, tf.squeeze(depth_pred), self.RT, self.intrinsic, ret_flows=False)
            print('warped pred', warped_pred)

            fake_tgt = tf.multiply(pixel_pred, mask_pred) + tf.multiply(warped_pred, 1-mask_pred)

        with tf.name_scope('loss'):
            self.eval_loss ={}

            depth_loss = tf.reduce_mean(tf.abs(self.tgt_image - warped_pred)) * self.loss_weight
            pixel_loss = tf.reduce_mean(tf.abs(self.tgt_image - pixel_pred)) * self.loss_weight
            mask_loss = tf.reduce_mean(tf.abs(self.tgt_image - fake_tgt)) * self.loss_weight

            self.total_loss = depth_loss + pixel_loss + mask_loss

            self.eval_loss['depth_loss'] = depth_loss
            self.eval_loss['pixel_loss'] = pixel_loss
            self.eval_loss['mask_loss'] = mask_loss
            self.eval_loss['total_loss'] = self.total_loss

        # Summaries
        tf.summary.image('src_image', self.deprocess_image(self.src_image))
        tf.summary.image('tgt_image', self.deprocess_image(self.tgt_image))

        tf.summary.image('fake_tgt_image', self.deprocess_image(fake_tgt))
        tf.summary.image('pixel_pred_image', self.deprocess_image(pixel_pred))
        tf.summary.image('warped_pred_image', warped_pred)
        tf.summary.scalar('total_loss', self.total_loss)


        # Define optimizers
        with tf.name_scope('train_optimizers'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1)
            train_vars = [var for var in tf.trainable_variables()]
            grads_and_vars = self.optimizer.compute_gradients(self.total_loss, var_list=train_vars)
            self.train_op = self.optimizer.apply_gradients(grads_and_vars)


    def set_configuration(self):
        # Check training variables
        with tf.name_scope("train_var"):
            all_vars = tf.trainable_variables()

            enc_var = [v for v in all_vars if 'Encoder' in v.op.name or 'encoder' in v.op.name]
            print(cyan('================== ENCODER ==================='))
            slim.model_analyzer.analyze_vars(enc_var, print_info=True)

            d_var = [v for v in all_vars if 'Depth' in v.op.name or 'depth' in v.op.name]
            print(cyan('================ DEPTH branch ================'))
            slim.model_analyzer.analyze_vars(d_var, print_info=True)

            p_var = [v for v in all_vars if 'Pixel' in v.op.name or 'pixel' in v.op.name]
            print(cyan('================ PIXEL branch ================'))
            slim.model_analyzer.analyze_vars(p_var, print_info=True)

            m_var = [v for v in all_vars if 'Mask' in v.op.name or 'mask' in v.op.name]
            print(cyan('================ MASK branch ================='))
            slim.model_analyzer.analyze_vars(m_var, print_info=True)

            self.train_summary_op = tf.summary.merge_all(key='train')

            # Set up the saver
            self.saver = tf.train.Saver(max_to_keep=100)
            self.pretrain_saver = tf.train.Saver(var_list=all_vars, max_to_keep=1)
            self.pretrain_saver_enc = tf.train.Saver(var_list=enc_var, max_to_keep=1)
            self.pretrain_saver_d = tf.train.Saver(var_list=d_var, max_to_keep=1)
            self.pretrain_saver_p = tf.train.Saver(var_list=p_var, max_to_keep=1)
            self.pretrain_saver_m = tf.train.Saver(var_list=m_var, max_to_keep=1)

        # Set up training session
        with tf.name_scope('train_session_config'):
            self.summary_writer = tf.summary.FileWriter(self.checkpoint_dir)

            self.supervisor = tf.train.Supervisor(logdir=self.checkpoint_dir, is_chief=True, saver=None, summary_op=None,
                                                  summary_writer=self.summary_writer, save_summaries_secs=300,
                                                  save_model_secs=600, global_step=self.global_step)

            session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True),
                                            device_count={'GPU':1})
            self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)

        # Reload the checkpoint if exist
        with tf.name_scope('train_checkpoint'):
            if self.continue_train:
                if self.checkpoint_dir is not None:
                    # checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
                    print(yellow('Restoring latest checkpoint path: {}'.format(self.checkpoint_dir)))
                    self.pretrain_saver.restore(self.session, self.checkpoint_dir)


    def train(self, input_data):
        max_steps = self.config.max_steps
        save_latest_freq = self.config.save_latest_freq
        summary_freq = self.config.summary_freq

        tf.logging.info('Training Starts!')
        for step in range(1, max_steps):
            # periodic interence (if needed implement here)
            # if step % test_sample_freq == 0:
            #     pass

            # Run the step
            glob_step, losses = self.run_single_step(input_data, step=step)

            # Save the summary
            if step % summary_freq == 0:
                self.summary_writer.add_summary(self.supervisor.summary_op, global_step=glob_step)
                #log_mesg = self.create_log_message()
                #tf.logging.info(log_mesg)
                pass

            # Save the latest model
            if step % save_latest_freq == 0:
                tf.logging.info(' [*] Saving checkpoint to %s...' % self.checkpoint_dir)
                self.saver.save(self.session, os.path.join(self.checkpoint_dir, 'model.latest'), global_step=step)

    def run_single_step(self, input_batch, step=None, is_train=True, kwarg=None):
        start_time = time.time()

        batch_chunk = self.session.run(input_batch)
        fetch = {
                    'train_summary':self.train_summary_op,
                    'global_step':self.global_step,
                    'train_op': self.train_op,
                    'Loss': self.eval_loss
                 }

        fetch_values = self.session.run(fetch, feed_dict=self.get_feed_dict(batch_chunk, step=step))

        end_time = time.time()
        global_step = fetch_values['global_step']
        losses = fetch_values['Loss']
        train_summary = fetch_values['train_summary']

        return global_step, losses


    def get_feed_dict(self, batch_chunk, step=None, is_training=True):
        tfvec = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor([0, 0, 0, 1], dtype=tf.float32), 0), 0)
        tfvec = tf.tile(tfvec, [self.batch_size, 1, 1])
        pose4x4 = tf.concat([batch_chunk['RT'], tfvec], axis =1)  # tgt to src pose
        inv_pose4x4 = tf.matrix_inverse(pose4x4)

        fd = { self.src_image: batch_chunk['B'],  # [B, h, w, c]
               self.tgt_image: batch_chunk['A'],  # [B, h, w, c]
               self.src_depth: batch_chunk['B_depth'],  # [B, h, w]
               self.tgt_depth: batch_chunk['A_depth'],  # [B, h, w]
               self.RT: pose4x4, # [B, 4, 4]
               self.inv_RT: inv_pose4x4,  # [B, 4, 4]
               self.intrinsic: batch_chunk['intrinsic'], # [B, 3, 3]
               self.step: step,
               self.is_train: is_training
             }
        return fd

    def deprocess_image(self, image):
        """Undo the preprocessing.

        Args:
          image: the input image in float with range [-1, 1]
        Returns:
          A new image converted to uint8 [0, 255]
        """
        image = (image + 1.) / 2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

