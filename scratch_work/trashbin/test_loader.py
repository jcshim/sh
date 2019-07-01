import os
import numpy as np
import h5py
import tensorflow as tf


# params
scene_class = 'kitti'

def make_pairs_with_near_frames(id, num_digit = 4, bound = 10):
    # The files (in reality they should be in the folder somewhere)
    filename = 'data_{}.hdf5'.format(scene_class)

    file = os.path.join('./'.format(scene_class), filename)

    print("Reading %s ...", file)
    data = h5py.File(file, 'r')
    print("Reading Done: %s", file)

    # preprocessing and datasets augmentation
    image_src = data[id]['image'][()] / 255. * 2 - 1
    pose_mat = np.append(data[id]['pose_matrix'][()], [0, 0, 0, 1]).reshape((4, 4))
    pose = data[id]['pose'][()]

    id_num = int(id[-num_digit:])
    random_num = np.random.randint(-bound, bound)
    id_target = id[:-num_digit] + str(id_num + random_num).zfill(num_digit)

    if id_target in data:
        image_tgt = data[id_target]['image'][()] / 255. * 2 - 1
        pose_tgt_mat = np.append(data[id]['pose_matrix'][()], [0, 0, 0, 1]).reshape((4, 4))

        transformation_mat = np.matmul(pose_tgt_mat, np.linalg.inv(pose_mat))
        inv_transformation_mat = np.linalg.inv(transformation_mat)

        RT = transformation_mat[:3]
        inv_RT = inv_transformation_mat[:3]

    return image_src, image_tgt, RT, inv_RT


def main():
    # Define the list of files (here file IDs)
    with open(os.path.join('./'.format(scene_class), 'id_train.txt'), 'r') as fp:
        list_files = [s.strip() for s in fp.readlines() if s]
        print('The number of training ids is {}'.format(len(list_files)))

    # The files (in reality they should be in the folder somewhere)
    filename = 'data_{}.hdf5'.format(scene_class)

    file = os.path.join('./'.format(scene_class), filename)

    print("Reading %s ...", file)
    data = h5py.File(file, 'r')
    print("Reading Done: %s", file)



    # create a dataset from filenames
    dataset = tf.data.Dataset.from_tensor_slices(list_files)
    dataset.map(make_pairs_with_near_frames, num_parallel_calls=8)
    i =0
    for d in dataset:
        print(d)
    print(i)


if __name__ == '__main__':
    tf.enable_eager_execution()
    main()



