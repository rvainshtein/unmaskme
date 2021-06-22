import matplotlib.pyplot as plt

from cyclegan.datasets.create_combined_mask_dataset import combine_dataset
from masktheface.mask_the_face import prepare_args, do_masking
from masktheface.utils.aux_functions import mask_image
import shutil
import os


def flatten_dir(destination, depth=None):
    """
    taken from https://stackoverflow.com/questions/17547273/flatten-complex-directory-structure-in-python
    """
    if not depth:
        depth = []
    for file_or_dir in os.listdir(os.path.join([destination] + depth, os.sep)):
        if os.path.isfile(file_or_dir):
            shutil.move(file_or_dir, destination)
        else:
            flatten_dir(destination, os.path.join(depth + [file_or_dir], os.sep))


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
data_root = 'data_root'
ffhq_root = 'data_root/ffhq_data'

# extract images from subdirs
flatten_dir(ffhq_root)

# create masked images
# masked images will be under '{unmasked_data_dir}_masked'
unmasked_data_dir = 'ffhq_data'
args = prepare_args(f'{unmasked_data_dir}', verbose=False)

do_masking(args)

# now we split train_test
masked_data_dir = ffhq_root + '_masked'
split_train_test(ffhq_root, masked_data_dir, ratio=0.8)

# create aligned dataset
combined_data_dir = os.path.join(data_root, 'data_combined')
combine_dataset(ffhq_root, masked_data_dir, combined_data_dir)
