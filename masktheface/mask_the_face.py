# Author: aqeelanwar
# Created: 27 April,2020, 10:22 PM
# Email: aqeel.anwar@gatech.edu

import argparse
import dlib
from sklearn.utils import Bunch

from masktheface.utils.aux_functions import *


def prepare_args(path, verbose=True, mask_type='random'):
    args = Bunch()
    args.path = path
    args.mask_type = mask_type
    args.pattern = ''
    args.pattern_weight = 0.5
    # args.color = "#0473e2"
    # args.color_weight = 0.5
    args.code = ''  # "cloth-masks/textures/check/check_4.jpg, cloth-#e54294, cloth-#ff0000, cloth, cloth-masks/textures/others/heart_1.png, cloth-masks/textures/fruits/pineapple.png, N95, surgical_blue, surgical_green"
    args.verbose = verbose
    args.write_original_image = False

    args.write_path = args.path + "_masked"
    # Set up dlib face detector and predictor
    args.detector = dlib.get_frontal_face_detector()
    path_to_dlib_model = "dlib_models/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(path_to_dlib_model):
        download_dlib_model()
    args.predictor = dlib.shape_predictor(path_to_dlib_model)
    # Extract data from code
    mask_code = "".join(args.code.split()).split(",")
    args.code_count = np.zeros(len(mask_code))
    args.mask_dict_of_dict = {}
    for i, entry in enumerate(mask_code):
        mask_dict = {}
        mask_color = ""
        mask_texture = ""
        mask_type = entry.split("-")[0]
        if len(entry.split("-")) == 2:
            mask_variation = entry.split("-")[1]
            if "#" in mask_variation:
                mask_color = mask_variation
            else:
                mask_texture = mask_variation
        mask_dict["type"] = mask_type
        mask_dict["color"] = mask_color
        mask_dict["texture"] = mask_texture
        args.mask_dict_of_dict[i] = mask_dict
    return args


def do_masking(args):
    # Check if path is file or directory or none
    is_directory, is_file, is_other = check_path(args.path)
    display_MaskTheFace()
    if is_directory:
        path, dirs, files = os.walk(args.path).__next__()
        file_count = len(files)
        dirs_count = len(dirs)
        if len(files) > 0:
            print_orderly("Masking image files", 60)

        # Process files in the directory if any
        for f in tqdm(files):
            image_path = path + "/" + f

            write_path = path + "_masked"
            if not os.path.isdir(write_path):
                os.makedirs(write_path)

            if is_image(image_path):
                # Proceed if file is image
                if args.verbose:
                    str_p = "Processing: " + image_path
                    tqdm.write(str_p)

                split_path = f.rsplit(".")
                masked_image, mask, mask_binary_array, original_image = mask_image(
                    image_path, args
                )
                for i in range(len(mask)):
                    w_path = (
                            write_path
                            + "/"
                            + split_path[0]
                            + "_"
                            + mask[i]
                            + "."
                            + split_path[1]
                    )
                    img = masked_image[i]
                    cv2.imwrite(w_path, img)

        print_orderly("Masking image directories", 60)

        # Process directories withing the path provided
        for d in tqdm(dirs):
            dir_path = args.path + "/" + d
            dir_write_path = args.write_path + "/" + d
            if not os.path.isdir(dir_write_path):
                os.makedirs(dir_write_path)
            _, _, files = os.walk(dir_path).__next__()

            # Process each files within subdirectory
            for f in files:
                image_path = dir_path + "/" + f
                if args.verbose:
                    str_p = "Processing: " + image_path
                    tqdm.write(str_p)
                write_path = dir_write_path
                if is_image(image_path):
                    # Proceed if file is image
                    split_path = f.rsplit(".")
                    masked_image, mask, mask_binary, original_image = mask_image(
                        image_path, args
                    )
                    for i in range(len(mask)):
                        w_path = (
                                write_path
                                + "/"
                                + split_path[0]
                                + "_"
                                + mask[i]
                                + "."
                                + split_path[1]
                        )
                        w_path_original = write_path + "/" + f
                        img = masked_image[i]
                        # Write the masked image
                        cv2.imwrite(w_path, img)
                        if args.write_original_image:
                            # Write the original image
                            cv2.imwrite(w_path_original, original_image)

                if args.verbose:
                    print(args.code_count)

    # Process if the path was a file
    elif is_file:
        print("Masking image file")
        image_path = args.path
        write_path = args.path.rsplit(".")[0]
        if is_image(image_path):
            # Proceed if file is image
            # masked_images, mask, mask_binary_array, original_image
            masked_image, mask, mask_binary_array, original_image = mask_image(
                image_path, args
            )
            for i in range(len(mask)):
                w_path = write_path + "_" + mask[i] + "." + args.path.rsplit(".")[1]
                img = masked_image[i]
                cv2.imwrite(w_path, img)
    else:
        print("Path is neither a valid file or a valid directory")
    print("Processing Done")


if __name__ == '__main__':

    # Command-line input setup
    args = prepare_args()

    do_masking(args)
