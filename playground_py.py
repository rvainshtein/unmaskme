import matplotlib.pyplot as plt
from masktheface.mask_the_face import prepare_args, do_masking
from masktheface.utils.aux_functions import mask_image

image_path = 'data'
args = prepare_args(f'{image_path}', verbose=False)
# masked_images, mask, mask_binary_array, original_image = \
#     mask_image(image_path, args)
#
# plt.figure()
# plt.imshow(masked_images[0])
# plt.show()
do_masking(args)
