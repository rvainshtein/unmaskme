import argparse
import os
import shutil

from cyclegan.datasets.create_combined_mask_dataset import combine_dataset
from masktheface.mask_the_face import prepare_args, do_masking


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


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ffhq-root', help='FFHQ facedataset images dir.', required=True)
    parser.add_argument('--flattened-dir', help='for extracting images from FFHQ sub dirs.')
    parser.add_argument('--masked-data-dir', help='for saving masked face images.')
    parser.add_argument('--combined-data-dir', help='for final result.', required=True)
    parser.add_argument('--train-test-ratio', help='percentage of train set from all dataset', default=0.8)

    args = parser.parse_args()
    if args.flattened_dir is None:
        args.flattened_dir = os.path.join(args.ffhq_root, 'flattened')
    if args.masked_data_dir is None:
        args.masked_data_dir = args.flattened_dir + '_masked'
    return args


if __name__ == '__main__':
    # define dirs
    args = get_args()

    # extract images from subdirs
    flatten_dir(args.ffhq_root, output_dir=args.flattened_dir)

    # create masked images
    # masked images will be under '{flattened_dir}_masked'
    args = prepare_args(args.flattened_dir, verbose=False)

    do_masking(args)

    # now we split train_test
    split_train_test(args.flattened_dir, args.masked_data_dir, ratio=args.train_test_ratio)

    # create aligned dataset
    combine_dataset(args.flattened_dir, args.masked_data_dir, args.combined_data_dir)
