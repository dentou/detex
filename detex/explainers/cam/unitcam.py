import torch
import numpy as np
from .ssd_extractor import ModelOutputs

class UnitCAM:
    """Unit Class Activation Mapping (UnitCAM)
    UnitCAM is the foundation for implementing all the CAMs
    Attributes:
    -------
        model: The wanna-be explained deep learning model for image classification
        num_classes: Total classes in the dataset
        use_cuda: Whether to use cuda
    """

    def __init__(self, model, baseline, num_classes, use_cuda):
        self.model = model
        self.baseline = baseline
        self.num_classes = num_classes
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

    def forward(self, input_features):
        """Forward pass
        Attributes:
        -------
            input_features: An image data input to the model
        Returns:
        -------
            Forward-pass result
        """
        return self.model(input_features)

    def _extract_features(self, input_features, print_out, bbox, index, zero_out=None):
        """Extract the feature maps of the targeted layer
        Attributes:
        -------
            input_features: An image input to the model
            bbox: Targeted bounding box
            index: Targeted output class
            print_out: Whether to print the maximum likelihood class
                (if index is set to None)
            zero_out: Whether to set the targeted module weights to 0
                (used in Ablation-CAM)
        Returns:
        -------
            features: The feature maps of the targeted layer
            detections: The forward-pass result
            raw_boxes_per_scale: All the detected boxes before being thresholded
            index: The targeted class index
            loc: Location of the targeted box
            overall_box_id: The box id
        """
        self.extractor = ModelOutputs(
            self.model, self.baseline,  self.num_classes, bbox
        )
        if self.cuda:
            if zero_out:
                features, detections, raw_boxes_per_scale, loc, overall_box_id = self.extractor(input_features.cuda(), bbox, zero_out)
            else:
                features, detections, raw_boxes_per_scale, loc, overall_box_id = self.extractor(input_features.cuda(), bbox)
        else:
            if zero_out:
                features, detections, raw_boxes_per_scale, loc, overall_box_id = self.extractor(input_features, bbox, zero_out)
            else:
                features, detections, raw_boxes_per_scale, loc, overall_box_id = self.extractor(input_features, bbox)

        if index is None:
            index = detections[0]['labels'][[torch.allclose(detections[0]['boxes'][idx], bbox) for idx in range(len(detections[0]['boxes']))]]
            if print_out:
                print(f"The index has the largest maximum likelihood is {index}")

        return features, detections, raw_boxes_per_scale, index, loc, overall_box_id
 
    @staticmethod
    def _cam_weighted_sum(cam, weights, target, ReLU=True):
        """Do linear combination between the defined weights and corresponding
        feature maps
        Attributes:
        -------
            cam: A placeholder for the final results
            weights: The weights computed based on the network output
            target: The targeted feature maps
        Returns:
        -------
            cam: The resulting weighted feature maps
        """
        try:
            for i, w in enumerate(weights):
                cam += w * target[i, :, :]
        except TypeError:
            cam += weights * target[0:1, :, :]

        if ReLU:
            cam = np.maximum(cam, 0)

        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-9)
        return cam

    def __call__(self, input_features, print_out, bbox, index=None):
        """Abstract methods for implementing in the sub classes
        Attributes:
        -------
            input_features: An image data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            bbox: Targeted bounding box
            index: Targeted output class
        """
        return NotImplementedError