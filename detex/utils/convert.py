import torch
import torchvision.transforms.functional as TF
import numpy as np

def tensorimg_to_npimg(img):
    """Convert tensor img to numpy img (uint8)

    Args:
        img (Tensor)

    Returns:
        numpy array channel last, uint8
    """
    if torch.is_tensor(img):
        return np.array(TF.to_pil_image(img))
    return img