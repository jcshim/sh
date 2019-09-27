from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np
import pickle
import pathlib, random

import matplotlib.pyplot as plt



__PATH__ = '/home/sokim/Documents/DATASETS/car_set3_part1/'

def check_data(data_path, vis=True):
    '''
        check data in the data_path
    :param data_path: path to the data (The data is assumed to be a pickle file)
    :param vis: Flag to visualize the loaded data
    :return:
    '''
    # open data
    data = pickle.load(open(data_path, 'rb'))


    A = data['A']
    B = data['B']
    A_depth = data['A_depth']
    B_depth = data['B_depth']
    A_pose = data['A_pose']
    B_pose = data['B_pose']
    RT = data['RT']

    print('A:{}, B:{}, A_depth:{}, B_depth:{}, RT:{}'.format(A.shape, B.shape, A_depth.shape, B_depth.shape, RT.shape))

    plt.ion()
    if vis:
        plt.subplot(221)
        plt.imshow(A)
        plt.subplot(222)
        plt.imshow(B)
        plt.subplot(223)
        plt.imshow(A_depth)
        plt.subplot(224)
        plt.imshow(B_depth)

        plt.show()
        plt.pause(2)


def _load_and_preprocess_data(data_path):

    # open data
    data = pickle.load(open(tfds.as_numpy(data_path), 'rb'))

    A = data['A']
    B = data['B']
    A_depth = data['A_depth']
    B_depth = data['B_depth']
    A_pose = data['A_pose']
    B_pose = data['B_pose']
    RT = data['RT']

    # Process the image : Reshape and normalize to [-1,1] range
    #A = tf.image.resize(A, [256, 256])
    A = 2*(A / 255.0) -1
    #B = tf.image.resize(B, [256, 256])
    B = 2*(B / 255.0) -1

    # Create an dictionary of the output
    data_out ={}
    data_out['A'] =A
    data_out['B'] =B
    data_out['A_depth'] = A_depth
    data_out['B_depth'] = B_depth
    data_out['A_pose'] = A_pose
    data_out['B_pose'] = B_pose
    data_out['RT'] = RT

    return A, B, A_depth, B_depth, RT

def _set_shapes(A, B, A_depth, B_depth, RT):
    A.set_shape([256, 256, 3])
    B.set_shape([256, 256, 3])
    A_depth.set_shape([256, 256])
    B_depth.set_shape([256, 256])
    RT.set_shape([3, 4])
    return A, B, A_depth, B_depth, RT

def vis_data(data):
    A_raw, B_raw, A_depth, B_depth, RT = data

    batch, _, _, _ = A_raw.shape

    plt.ion()
    for i in range(batch):
        print(i, A_raw[i].shape)
        A = (A_raw[i] + 1) / 2 * 255
        B = (B_raw[i] + 1) / 2 * 255
        plt.subplot(221)
        plt.imshow(A.astype(np.uint8))
        plt.subplot(222)
        plt.imshow(B.astype(np.uint8))
        plt.subplot(223)
        plt.imshow(A_depth[i])
        plt.subplot(224)
        plt.imshow(B_depth[i])

        plt.show()
        plt.pause(2)



if __name__ == '__main__':
    # tf.enable_eager_execution()

    # Retrieve the data from the PATH directory
    data_root = pathlib.Path(__PATH__)

    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    image_count = len(all_image_paths)
    print('The total image number is {}'.format(image_count))

    for n in range(1):
        data_path =  random.choice(all_image_paths)
        check_data(data_path, vis=False)

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    print(path_ds)
    print('shape: ', repr(path_ds.output_shapes))
    print('type: ', path_ds.output_types)
    print()

    image_ds = path_ds.map(lambda all_image_paths: tf.py_function(_load_and_preprocess_data,inp=[all_image_paths],
                                                    Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]))
    print(image_ds)
    print('shape: ', repr(image_ds.output_shapes))
    print('type: ', image_ds.output_types)
    print()

    ds = image_ds.map(_set_shapes)
    print(ds)
    print('shape: ', repr(ds.output_shapes))
    print('type: ', ds.output_types)
    print()

    BATCH_SIZE = 5
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # Setting a shuffle buffer size as large as the dataset ensures that the data is
    # completely shuffled.
    ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=200))
    ds = ds.batch(batch_size=BATCH_SIZE)
    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    ds = ds.prefetch(buffer_size=1)
    print(ds)
    print('shape: ', repr(ds.output_shapes))
    print('type: ', ds.output_types)
    print()

    iterator = ds.make_one_shot_iterator()
    example = iterator.get_next()

    with tf.Session() as sess:
        for i in range(2):
            # sess.run(tf.global_variables_initializer())
            # sess.run(iterator.initializer, feed_dict = {placeholder_X: A})
            try:
                # while True:
                print('This iteration {}'.format(i))
                val = sess.run(example)
                # print(val)
                vis_data(val)
            except tf.errors.OutOfRangeError:
                pass

