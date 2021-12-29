import torch
import numpy as np
from .convert import tensorimg_to_npimg
from skimage import segmentation

def segment(img):
    img = tensorimg_to_npimg(img)
    spixel_labels = segmentation.slic(
        img,
        n_segments=256,
        sigma=1.4,
        compactness=10,
        convert2lab=True,
        slic_zero=True,
        multichannel=True,
    )  # a higher value of compactness leads to squared regions, a higher value of sigma leads to rounded delimitations
    return spixel_labels  # , skseg.mark_boundaries(img, spixel_labels)