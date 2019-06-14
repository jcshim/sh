from __future__ import division
import tensorflow as tf
import time, os

import datasets.kitti_loader as dataset
from datasets.input_ops import create_input_ops
from datasets.car_loader import CarDataLoader

from models.test_model2 import Model

flags = tf.app.flags
flags.DEFINE_string('checkpoint_dir', './checkpoints','Location to save the models.')
flags.DEFINE_string('experiment_name', '', 'Name for the experiment to run. It decides where to store samples and models.')
flags.DEFINE_string('data', 'car', 'Type of the datasets. [car | kitti]')

flags.DEFINE_integer('batch_size', 4, 'Input batch size')

flags.DEFINE_integer('max_steps', 10, 'Maximum number of training steps.(epoch)')
flags.DEFINE_integer('summary_freq', 50, 'Logging frequency.')
flags.DEFINE_integer('save_latest_freq', 2000, 'Frequency with which to save the model (overwrites previous model).')

flags.DEFINE_float('learning_rate', 0.00006, 'Initial learning rate for Adam optimizer')
flags.DEFINE_float('beta1', 0.5, 'beta1 hyperparameter for Adam optimizer.')

flags.DEFINE_boolean('continue_train', False, 'Continue training from previous checkpoint.')
FLAGS = flags.FLAGS

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    # tf.set_random_seed(FLAGS.random_seed)

    # Create checkpoint directory
    FLAGS.checkpoint_dir += '/{}/'.format(FLAGS.experiment_name)
    if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    ##==== for kitti loader ====#
    # # Set up datasets loader
    # dataset_train, dataset_test = dataset.create_default_splits()
    #
    # # Set input batches
    # _, batch_train = create_input_ops(dataset_train, FLAGS.batch_size)
    # # _, batch_test = create_input_ops(dataset_test, FLAGS.batch_size)
    # print(batch_train)
    #===========================#

    # Set up datasets loader
    data_loader = CarDataLoader(FLAGS)
    train_batch = data_loader.sample_batch()
    print(train_batch)

    # Set up network
    model = Model(FLAGS)
    train_op = model.build_train_graph()
    model.set_configuration()

    # Start training
    model.train(train_op)

    # Stereo magnificiatnce
  #     model = MPI()
  #     train_op = model.build_train_graph(
  #         train_batch, FLAGS.min_depth, FLAGS.max_depth, FLAGS.num_psv_planes,
  #         FLAGS.num_mpi_planes, FLAGS.which_color_pred, FLAGS.which_loss,
  #         FLAGS.learning_rate, FLAGS.beta1, FLAGS.vgg_model_file)
  #     model.train(train_op, FLAGS.checkpoint_dir, FLAGS.continue_train,
  #                 FLAGS.summary_freq, FLAGS.save_latest_freq, FLAGS.max_steps)

if __name__ == '__main__':
    tf.app.run()