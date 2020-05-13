# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 10:30:19 2019

@author: YoupSuurmeijer
"""
import numpy as np
from utils import DEFAULT_SETTINGS
import cv2
from PIL import Image
from deeplab import *

class BackgroundRemover(object):
    """
    Create Background Remover Object
    settings: dict containing the background remover settings
    settings['MASK_COLOR']: Color for the masking of the image default is white to create effect of disappearing bg
    settings['MASK_DILATE_ITER']: Expands the mask for a 'roomier' fit around the person (used for smoothing mask edge)
    settings['MASK_ERODE_ITER']: Contracts the mask for a 'tighter' fit around the person (used for smoothing mask edge)
    settings['MASK_BLUR']: Settings to be used for the Gaussian blurring of the mask (used for smoothing mask edge)
    settings['MASK_THRESHOLD']: Gaussian smoothing results in non-binary mask, threshold value used as cutoff to
                                convert back to binary mask
    """
    def __init__(self, model, settings=DEFAULT_SETTINGS['BackgroundRemover']):
        self.model = DeepLabModel(model) # models.segmentation.fcn_resnet101(pretrained=True).eval()
        self.settings = settings

    def make_background_transparant(self, image):
        """
        Main function to create an image with transparant background from the input image

        Parameters
        ----------
        image : open-cv format image

        Returns
        -------
        open-cv formatted image with background removed

        """

        # Resize image to required model input size

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)

        # Run the prediction function
        segmentation = self.model.run(im_pil) #self.segment(self.model, im_pil)

        # Generate mask binary mask from segmentation (class 15 is a pixel containing a person)
        mask = np.float32(segmentation[1])
        mask[mask != 15] = 0
        mask[mask == 15] = 1

        # Resize mask again to original image size so that we can preserve input image quality
        mask = cv2.resize(mask, (image.shape[1],image.shape[0]), interpolation = cv2.INTER_NEAREST)

        # Apply dilate, erode and blur functions to smooth the edges of the mask and create a tighter/looser fit
        # Note that the Gaussian blur means the mask is no longer binary (thus we use a threshold later)
        mask = cv2.dilate(mask, None, iterations=self.settings['MASK_DILATE_ITER'])
        mask = cv2.erode(mask, None, iterations=self.settings['MASK_ERODE_ITER'])
        mask = cv2.GaussianBlur(mask, (self.settings['MASK_BLUR'], self.settings['MASK_BLUR']), 0)

        # Set all pixel values of the input image that are below the mask threshold to the mask color
        image_array = np.float32(image)
        subset = np.less(mask, self.settings['MASK_THRESHOLD'])
        image_array[:,:,:3][subset] = self.settings['MASK_COLOR']

        return image_array


if __name__=='__main__':
    import os
    import glob
    from line_profiler import LineProfiler

    #Set images directory
    root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    image_dir = r'images\test_set_21_01\TEST1VIPO'
    search_dir = os.path.join(root_dir, image_dir)
    out_dir = os.path.join(root_dir, 'test\output')
    os.chdir(search_dir)

    bg_remover = BackgroundRemover(model=os.path.join(root_dir, r'models\\deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz'))

    lp = LineProfiler()
    lp_wrapper = lp(bg_remover.make_background_transparant)

    for file in glob.glob("*.jpg")[:20]:
        print(file)
        image = cv2.imread(file)
        masked_image = lp_wrapper(image)
        cv2.imwrite(os.path.join(out_dir, 'input_'+file), image)
        cv2.imwrite(os.path.join(out_dir, 'output_'+file), masked_image)


    lp.print_stats(output_unit=1e-03)
    lp.dump_stats(os.path.join(out_dir, 'line_profile_bg_remover.pickle'))
