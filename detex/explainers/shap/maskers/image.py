import shap
import numpy as np
import cv2

class BetterImageMasker(shap.maskers.Image):
    def __call__(self, mask, x):
        if np.prod(x.shape) != np.prod(self.input_shape):
            raise Exception("The length of the image to be masked must match the shape given in the " + \
                            "ImageMasker contructor: "+" * ".join([str(i) for i in x.shape])+ \
                            " != "+" * ".join([str(i) for i in self.input_shape]))

        # unwrap single element lists (which are how single input models look in multi-input format)
        if isinstance(x, list) and len(x) == 1:
            x = x[0]

        # we preserve flattend inputs as flattened and full-shaped inputs as their original shape
        in_shape = x.shape
        if len(x.shape) > 1:
            x = x.ravel()

        # if mask is not given then we mask the whole image
        if mask is None:
            mask = np.zeros(np.prod(x.shape), dtype=np.bool)

        if isinstance(self.mask_value, str):
            if self.blur_kernel is not None:
                if self.last_xid != id(x):
                    self._blur_value_cache = cv2.blur(x.reshape(self.input_shape), self.blur_kernel).ravel()
                    self.last_xid = id(x)
                out = x.copy()
                out[~mask] = self._blur_value_cache[~mask]

            elif self.mask_value == "inpaint_telea":
                out = self.inpaint(x, ~mask, "INPAINT_TELEA")
            elif self.mask_value == "inpaint_ns":
                out = self.inpaint(x, ~mask, "INPAINT_NS")
        else:
            out = x.copy()
            out[~mask.flatten()] = self.mask_value[~mask.flatten()]

        return (out.reshape(1, *in_shape),)