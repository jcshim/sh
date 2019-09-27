from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
import pathlib, random
import numpy as np

import matplotlib.pyplot as plt


__PATH__ = '/home/sokim/Documents/DATASETS/car_set3_part1/'


class CarDataLoader(object):
    """Loader for synthesized car data"""
    def __init__(self, flag):
        self.batch_size = flag.batch_size
        self.image_height = 256
        self.image_width = 256
        self.buffer_size = 100

    def sample_batch(self):
        """Samples a batch of examples for training / testing.

        Returns:
          A batch of examples.
        """
        batch_example = self.load_data()
        iterator = batch_example.make_one_shot_iterator()
        return self._set_dictionary(iterator.get_next())

    def _set_dictionary(self, batch_sample):
        """
            Make batch into dictionary format
        :param batch_sample: batch sample grabbed from each iteration
        :return:
        """
        A, B, A_depth, B_depth, RT, INTRINSIC = batch_sample

        out_example = {}
        out_example['A'] = A
        out_example['B'] = B
        out_example['A_depth'] = A_depth
        out_example['B_depth'] = B_depth
        out_example['RT'] = RT
        out_example['intrinsic'] = INTRINSIC
        # examples['intrinsics'].set_shape([b, 3, 3])

        return out_example

    def _load_and_preprocess_data(self, data_path):

        # open data
        data = pickle.load(open(tfds.as_numpy(data_path), 'rb'))

        A = data['A']
        B = data['B']
        A_depth = data['A_depth']
        B_depth = data['B_depth']
        A_pose = data['A_pose']
        B_pose = data['B_pose']
        RT = data['RT']
        intrinsic = self._get_car_intrinsic()

        # Process the image : Reshape and normalize to [-1,1] range
        #A = tf.image.resize(A, [256, 256])
        A = 2 * (A / 255.0) - 1
        #B = tf.image.resize(B, [256, 256])
        B = 2 * (B / 255.0) - 1

        # Create an dictionary of the output
        # data_out ={}
        # data_out['A'] =A
        # data_out['B'] =B
        # data_out['A_depth'] = A_depth
        # data_out['B_depth'] = B_depth
        # data_out['A_pose'] = A_pose
        # data_out['B_pose'] = B_pose
        # data_out['RT'] = RT

        return A, B, A_depth, B_depth, RT, intrinsic

    def _set_shapes(self, A, B, A_depth, B_depth, RT, INTRINSIC):
        A.set_shape([256, 256, 3])
        B.set_shape([256, 256, 3])
        A_depth.set_shape([256, 256])
        B_depth.set_shape([256, 256])
        RT.set_shape([3, 4])
        INTRINSIC.set_shape([3,3])
        return A, B, A_depth, B_depth, RT, INTRINSIC

    def load_data(self):
        # Retrieve the data from the PATH directory
        data_root = pathlib.Path(__PATH__)

        all_image_paths = list(data_root.glob('*'))
        all_image_paths = [str(path) for path in all_image_paths]
        random.shuffle(all_image_paths)
        image_count = len(all_image_paths)
        print('The total image number is {}'.format(image_count))

        path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        image_ds = path_ds.map(lambda all_image_paths: tf.py_function(self._load_and_preprocess_data,inp=[all_image_paths],
                                                        Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]))
        ds = image_ds.map(self._set_shapes)

        # Setting a shuffle buffer size as large as the dataset ensures that the data is
        # completely shuffled.
        ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=self.buffer_size))
        ds = ds.batch(batch_size=self.batch_size)
        # `prefetch` lets the dataset fetch batches, in the background while the model is training.
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds

    def _get_car_intrinsic(self):
        # Intrinsics for Depth
        sensor_size, focal_length = 32, 60
        f = 256 / sensor_size * focal_length
        c = 256 / 2.
        intrinsics = np.array([f, 0, c, 0, f, c, 0, 0, 1]).reshape((3, 3))

        return intrinsics
