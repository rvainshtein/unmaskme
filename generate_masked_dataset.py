import matplotlib.pyplot as plt

from cyclegan.datasets.create_combined_mask_dataset import combine_dataset
from masktheface.mask_the_face import prepare_args, do_masking
from masktheface.utils.aux_functions import mask_image
import shutil
import os


def flatten_dir(source, depth=None, output_dir=None):
    """
    taken from https://stackoverflow.com/questions/17547273/flatten-complex-directory-structure-in-python
    with bugfixes
    """
    if output_dir is None:
        output_dir = source
    else:
        os.makedirs(output_dir, exist_ok=True)
    if not depth:
        depth = ''
    curr_dir = os.path.join(source, depth)
    for file_or_dir in os.listdir(curr_dir):
        curr_path = os.path.join(curr_dir, file_or_dir)
        if os.path.isfile(curr_path):
            shutil.move(curr_path, output_dir)
        else:
            flatten_dir(source, os.path.join(depth, file_or_dir))
    for file_or_dir in os.listdir(curr_dir):
        curr_path = os.path.join(curr_dir, file_or_dir)
        if os.path.isdir(curr_path):
            os.rmdir(curr_path)


def split_train_test(unmasked_data_dir, masked_data_dir, ratio=0.8):
    split_train_test_subdir(ratio, unmasked_data_dir)
    split_train_test_subdir(ratio, masked_data_dir)


def split_train_test_subdir(ratio, data_subdir):
    # make dirs
    os.makedirs(os.path.join(data_subdir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_subdir, 'test'), exist_ok=True)

    subdir_files = os.listdir(data_subdir)
    # no need for random since they are already random
    train_idx = int(ratio * len(subdir_files))
    train_files = subdir_files[:train_idx]
    test_files = subdir_files[train_idx:]

    move_subset_files(train_files, data_subdir, 'train')
    move_subset_files(test_files, data_subdir, 'test')


def move_subset_files(file_names, dir, subset):
    for f in file_names:
        orig_file_path = os.path.join(dir, f)
        if os.path.isfile(orig_file_path):
            shutil.move(orig_file_path, os.path.join(dir, subset, f))


# define dirs
data_root = r'data_root\output_mock'
ffhq_root = r'data_root\mock'

# extract images from subdirs
flattened_dir = r'data_root\flattened'
flatten_dir(ffhq_root, output_dir=flattened_dir)

# create masked images
# masked images will be under '{flattened_dir}_masked'
args = prepare_args(flattened_dir, verbose=False)

do_masking(args)

# now we split train_test
masked_data_dir = flattened_dir + '_masked'
split_train_test(flattened_dir, masked_data_dir, ratio=0.8)

# create aligned dataset
combined_data_dir = os.path.join(data_root, 'data_combined')
combine_dataset(flattened_dir, masked_data_dir, combined_data_dir)
