import torch
from torch.nn import functional as F
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import boxes as box_ops
import numpy as np
from .unitcam import UnitCAM
from .ssd_extractor import SSD300VGG16FeatureExtractor


class GradCAM(UnitCAM):
    """The implementation of Grad-CAM for object detection
    CNN-based deep learning models
    Based on the paper:
        Selvaraju, R. R., Cogswell, M.,
        Das, A., Vedantam, R., Parikh,
        D., & Batra, D. (2017). Grad-cam: Visual explanations from deep networks
        via gradient-based localization. In Proceedings of the
        IEEE international conference on computer vision (pp. 618-626).
    Implementation adapted from:
        https://github.com/jacobgil/pytorch-grad-cam/blob/bf27469f5b3accf9535e04e52106e3f77f5e9cf5/gradcam.py#L31
    Attributes:
    -------
        model: The wanna-be explained deep learning model for object detection
        num_classes: Total classes in the dataset
        use_cuda: Whether to use cuda
    """

    def __init__(self, model, baseline, num_classes, use_cuda):
        super().__init__(model, baseline, num_classes, use_cuda)
        self.grads_val = None
        self.target = None

    def _calculate_gradients(self, input_features, print_out, bbox, index):
        """Implemented method when CAM is called on a given input and its targeted
        index
        Attributes:
        -------
            input_features: An image data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            bbox: Targeted bounding box
            index: Targeted output class
        """
        features, output, raw_boxes_per_scale, index, loc, overall_box_id = self._extract_features(
            input_features, print_out, bbox, index
        )
        self.model.zero_grad()

        bboxes_per_scale = raw_boxes_per_scale['bbox_regression']
        labels_per_scale = raw_boxes_per_scale['cls_logits']

        try:
            self.one_hot_cls = torch.zeros_like(labels_per_scale[loc[0]], dtype=torch.float32)
            self.one_hot_cls[0, loc[1], index] = 1
            self.one_hot_cls.requires_grad_(True)
            self.one_hot_box = torch.zeros_like(bboxes_per_scale[loc[0]], dtype=torch.float32)
            self.one_hot_box[0, loc[1]] = 1
            self.one_hot_box.requires_grad_(True)
        except UnboundLocalError:
            raise ValueError("The target bbox does not exist.")
        
        if self.cuda:
            self.one_hot_cls = torch.sum(self.one_hot_cls.cuda() * labels_per_scale[loc[0]])
            self.one_hot_box = torch.sum(self.one_hot_box.cuda() * bboxes_per_scale[loc[0]])
        else:
            self.one_hot_cls = torch.sum(self.one_hot_cls * labels_per_scale[loc[0]])
            self.one_hot_box = torch.sum(self.one_hot_box * bboxes_per_scale[loc[0]])

        self.one_hot_cls.backward(retain_graph=True)
        self.one_hot_box.backward(retain_graph=True)

        self.grads_val_cls = self.extractor.get_gradients()[0].cpu().data
        self.grads_val_box = self.extractor.get_gradients()[1].cpu().data

        self.target_cls = features[0]
        self.target_cls = self.target_cls.cpu().data.numpy()[0, :]

        self.target_box = features[0]
        self.target_box = self.target_box.cpu().data.numpy()[0, :]

        return output, index, overall_box_id

    def __call__(self, input_features, print_out, bbox, index=None):
        """Implemented method when CAM is called on a given input and its targeted
        index
        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class
        Returns:
        -------
            cam: The resulting weighted feature maps
        """
        if index is not None and print_out == True:
            print_out = False

        output, index, overall_box_id = self._calculate_gradients(input_features, print_out, bbox, index)

        cam_cls, cam_box, weights_cls, weights_box = self._map_gradients()
        assert (
            weights_cls.shape[0] == self.target_cls.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        assert (
            weights_box.shape[0] == self.target_box.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        cam_cls = self._cam_weighted_sum(cam_cls, weights_cls, self.target_cls)
        cam_box = self._cam_weighted_sum(cam_box, weights_box, self.target_box)

        return cam_cls, cam_box, output, index, overall_box_id

class XGradCAM(GradCAM):
    """The implementation of XGrad-CAM for object detection
    CNN-based deep learning models
    Based on the paper:
        Fu, R., Hu, Q., Dong, X., Guo, Y., Gao, Y., & Li, B. (2020). Axiom-based
        grad-cam: Towards accurate visualization and explanation of cnns.
        arXiv preprint arXiv:2008.02312.
    Implementation adapted from:
        https://github.com/Fu0511/XGrad-CAM/blob/main/XGrad-CAM.py
    Classification data and the corresponding CNN-based models
    Attributes:
    -------
        model: The wanna-be explained deep learning model for object detection
        num_classes: Total classes in the dataset
        use_cuda: Whether to use cuda
    """

    def __init__(self, model, baseline, num_classes, use_cuda):
        super().__init__(model, baseline, num_classes, use_cuda)

    def _map_gradients(self):
        """Caculate weights based on the gradients corresponding to the extracting layer
        via global average pooling
        Returns:
        -------
            cam_cls: The placeholder for resulting weighted feature maps explaining classes
            cam_box: The placeholder for resulting weighted feature maps explaining boxes
            weights_cls: The weights corresponding to the extracting feature maps explaining classes
            weights_box: The weights corresponding to the extracting feature maps explaining boxes
        """
        self.target_cls = self.target_cls.reshape(*self.grads_val_cls.numpy()[0, :].shape)
        self.target_box = self.target_box.reshape(*self.grads_val_box.numpy()[0, :].shape)
        weights_cls = np.sum(self.grads_val_cls.numpy()[0,:] * self.target_cls, axis=(1, 2))
        weights_cls = weights_cls / (np.sum(self.target_cls, axis=(1, 2)) + 1e-6)
        weights_box = np.sum(self.grads_val_box.numpy()[0,:] * self.target_box, axis=(1, 2))
        weights_box = weights_box / (np.sum(self.target_box, axis=(1, 2)) + 1e-6)
        cam_cls = np.zeros(self.target_cls.shape[1:], dtype=np.float32)
        cam_box = np.zeros(self.target_box.shape[1:], dtype=np.float32)

        return cam_cls, cam_box, weights_cls, weights_box

class GradCAMPlusPlus(GradCAM):
    """The implementation of Grad-CAM++ for object detection
    CNN-based deep learning models
    Based on the paper:
        Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N.
        (2018, March). Grad-cam++: Generalized gradient-based visual explanations
        for deep convolutional networks.
        In 2018 IEEE Winter Conference on Applications of Computer Vision (WACV)
        (pp. 839-847). IEEE.
    Implementation adapted from:
        https://github.com/adityac94/Grad_CAM_plus_plus/blob/4a9faf6ac61ef0c56e19b88d8560b81cd62c5017/misc/utils.py#L51
    Classification data and the corresponding CNN-based models
    Attributes:
    -------
        model: The wanna-be explained deep learning model for object detection
        num_classes: Total classes in the dataset
        use_cuda: Whether to use cuda
    """

    def __init__(self, model, baseline, num_classes, use_cuda):
        super().__init__(model, baseline, num_classes, use_cuda)
        self.alphas = None
        self.one_hot = None

    @staticmethod
    def _compute_second_derivative(one_hot, target):
        """Second Derivative
        Attributes:
        -------
            one_hot: Targeted index output
            target: Targeted module output
        Returns:
        -------
            second_derivative: The second derivative of the output
        """
        second_derivative = torch.exp(one_hot.detach().cpu()) * target

        return second_derivative

    @staticmethod
    def _compute_third_derivative(one_hot, target):
        """Third Derivative
        Attributes:
        -------
            one_hot: Targeted index output
            target: Targeted module output
        Returns:
        -------
            third_derivative: The third derivative of the output
        """
        third_derivative = torch.exp(one_hot.detach().cpu()) * target * target

        return third_derivative

    @staticmethod
    def _compute_global_sum(one_hot):
        """Global Sum
        Attributes:
        -------
            one_hot: Targeted index output
        Returns:
        -------
            global_sum: Collapsed sum from the input
        """

        global_sum = np.sum(one_hot.detach().cpu().numpy(), axis=0)

        return global_sum

    def _extract_higher_level_gradient(
        self, global_sum, second_derivative, third_derivative
    ):
        """Alpha calculation
        Calculate alpha based on high derivatives and global sum
        Attributes:
        -------
            global_sum: Collapsed sum from the input
            second_derivative: The second derivative of the output
            third_derivative: The third derivative of the output
        """
        alpha_num = second_derivative.numpy()
        alpha_denom = (
            second_derivative.numpy() * 2.0 + third_derivative.numpy() * global_sum
        )
        alpha_denom = np.where(
            alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape)
        )
        alphas = alpha_num / alpha_denom

        return alphas

    def _map_gradients(self):
        """Caculate weights based on the gradients corresponding to the extracting layer
        via global average pooling
        Returns:
        -------
            cam_cls: The placeholder for resulting weighted feature maps explaining classes
            cam_box: The placeholder for resulting weighted feature maps explaining boxes
            weights_cls: The weights corresponding to the extracting feature maps explaining classes
            weights_box: The weights corresponding to the extracting feature maps explaining boxes
        """
        self.alphas_cls = self.alphas_cls.reshape(*self.grads_val_cls.numpy()[0, :].shape)
        self.alphas_box = self.alphas_box.reshape(*self.grads_val_box.numpy()[0, :].shape)
        self.target_cls = self.target_cls.reshape(*self.grads_val_cls.numpy()[0, :].shape)
        self.target_box = self.target_box.reshape(*self.grads_val_box.numpy()[0, :].shape)
        weights_cls = np.sum(F.relu(self.grads_val_cls).numpy()[0, :] * self.alphas_cls, axis=(1, 2))
        weights_box = np.sum(F.relu(self.grads_val_box).numpy()[0, :] * self.alphas_box, axis=(1, 2))
        cam_cls = np.zeros(self.target_cls.shape[1:], dtype=np.float32)
        cam_box = np.zeros(self.target_box.shape[1:], dtype=np.float32)

        return cam_cls, cam_box, weights_cls, weights_box

    def __call__(self, input_features, print_out, bbox, index=None):
        """Implemented method when CAM is called on a given input and its targeted
        index
        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            bbox: Targeted bounding box
            index: Targeted output class
        Returns:
        -------
            cam_cls: The resulting weighted feature maps explaining classes
            cam_box: The resulting weighted feature maps explaining boxes
            output: The detection results
            index: The targeted class label
            overall_box_id: The id of the bounding box among the box list
        """
        if index is not None and print_out == True:
            print_out = False

        output, index, overall_box_id = self._calculate_gradients(input_features, print_out, bbox, index)
        second_derivative_cls = self._compute_second_derivative(self.one_hot_cls, self.target_cls)
        third_derivative_cls = self._compute_third_derivative(self.one_hot_cls, self.target_cls)
        global_sum_cls = self._compute_global_sum(self.one_hot_cls)
        self.alphas_cls = self._extract_higher_level_gradient(
            global_sum_cls, second_derivative_cls, third_derivative_cls
        )

        second_derivative_box = self._compute_second_derivative(self.one_hot_box, self.target_box)
        third_derivative_box = self._compute_third_derivative(self.one_hot_box, self.target_box)
        global_sum_box = self._compute_global_sum(self.one_hot_box)
        self.alphas_box = self._extract_higher_level_gradient(
            global_sum_box, second_derivative_box, third_derivative_box
        )

        cam_cls, cam_box, weights_cls, weights_box = self._map_gradients()
        assert (
            weights_cls.shape[0] == self.target_cls.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        assert (
            weights_box.shape[0] == self.target_box.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        cam_cls = self._cam_weighted_sum(cam_cls, weights_cls, self.target_cls)
        cam_box = self._cam_weighted_sum(cam_box, weights_box, self.target_box)

        return cam_cls, cam_box, output, index, overall_box_id