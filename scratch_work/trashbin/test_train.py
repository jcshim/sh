from __future__ import division
import tensorflow as tf

import datasets.kitti_loader as dataset
from datasets.input_ops import create_input_ops

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    # tf.set_random_seed(FLAGS.random_seed)
    # FLAGS.checkpoint_dir += '/%s/' % FLAGS.experiment_name
    # if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
    #     tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    # Set up datasets loader

    dataset_train, dataset_test = dataset.create_default_splits()
    print(len(dataset_train))

    _, batch_train = create_input_ops(dataset_train, FLAGS.batch_size)



    # Set up network


    # Start training
