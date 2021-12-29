from . import BaseWrapper
import torch
import torch.nn.functional as F
import warnings

from torchvision.ops import boxes as box_ops
import torchvision.transforms.functional as TF


from collections import OrderedDict
from torch import nn, Tensor
from typing import Any, Dict, List, Optional, Tuple, Callable

from torchvision.models.detection.transform import (
    resize_boxes,
    paste_masks_in_image,
    resize_keypoints,
)
import types

import numpy as np


class SSDWrapper(BaseWrapper):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

        self.model.postprocess_detections = types.MethodType(
            SSDWrapper._postprocess_detections, model
        )
        self.model.transform.postprocess = types.MethodType(
            SSDWrapper._transform_postprocess, model.transform
        )

    def make_blackbox(self, engine, box_id, box_attr, device) -> "Callable":
        """Make single-input forward func for explainer

        Args:
            engine: "captum" or "shap"
            box_id: index of box in raw_boxes
            box_attr: index of attributes in raw_boxes. raw_boxes has shape (M, 4+C) where C is #classes

        """

        @torch.no_grad()
        def blackbox_shap(imgs):
            """For shap.Explainer

            Args:
                imgs: 4D ndarray (N, H, W, C)

            Returns:
                1D ndarray with length N (one score for each image in batch)
            """
            imgs = [TF.to_tensor(img).to(device) for img in imgs]
            dets = self.forward_raw(imgs)
            dets = np.array(
                [
                    det["raw_boxes"][box_id, box_attr].detach().cpu().numpy()
                    for det in dets
                ]
            )
            return dets

        @torch.no_grad()
        def blackbox_captum(imgs):
            """For shap.Explainer

            Args:
                imgs: 4D tensor (N, C, H, W)

            Returns:
                1D tensor with length N (one score for each image in batch)
            """
            dets = self.forward_raw(imgs.to(device))
            dets = torch.cat(
                [det["raw_boxes"][box_id, box_attr].unsqueeze(0) for det in dets]
            )
            return dets

        if engine == "shap":
            return blackbox_shap
        elif engine == "captum":
            return blackbox_captum
        else:
            raise ValueError(f"engine={engine} not supported")

    def forward(
        self,
        images: "List[Tensor]",
        targets: "Optional[List[Dict[str, Tensor]]]" = None,
        postprocess: bool = True,
    ):
        if postprocess:
            return self.model.forward(images, targets)
        else:
            return self.forward_raw(images, targets)

    def forward_raw(self, images: "List[Tensor]"):
        if self.training:
            raise ValueError("forward_raw does not support training")

        # get the original image sizes
        original_image_sizes = []  # List[Tuple[int, int]]
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        targets = None
        images, targets = self.model.transform(images, targets)

        # get the features from the backbone
        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # compute the ssd heads outputs using the features
        head_outputs = self.model.head(features)

        # create the set of anchors
        anchors = self.model.anchor_generator(images, features)

        losses = {}
        detections = []  # List[Dict[str, Tensor]]

        bbox_regression = head_outputs["bbox_regression"]
        pred_scores = F.softmax(head_outputs["cls_logits"], dim=-1)

        detections = []  # List[Dict[str, Tensor]]

        for boxes, scores, anchors, image_shape in zip(
            bbox_regression, pred_scores, anchors, images.image_sizes
        ):
            boxes = self.model.box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            raw_boxes = torch.cat((boxes, scores), dim=1)

            detections.append(
                {"raw_boxes": raw_boxes,}
            )

        for i, (pred, im_s, o_im_s) in enumerate(
            zip(detections, images.image_sizes, original_image_sizes)
        ):

            detections[i]["raw_boxes"][:, :4] = resize_boxes(
                detections[i]["raw_boxes"][:, :4], im_s, o_im_s
            )

        if torch.jit.is_scripting():
            if not self.model._has_warned:
                warnings.warn(
                    "SSD always returns a (Losses, Detections) tuple in scripting"
                )
                self.model._has_warned = True
            return losses, detections
        return self.model.eager_outputs(losses, detections)

    @staticmethod
    def _postprocess_detections(
        self,
        head_outputs: "Dict[str, Tensor]",
        image_anchors: "List[Tensor]",
        image_shapes: "List[Tuple[int, int]]",
    ) -> "List[Dict[str, Tensor]]":

        bbox_regression = head_outputs["bbox_regression"]
        pred_scores = F.softmax(head_outputs["cls_logits"], dim=-1)

        num_classes = pred_scores.size(-1)
        device = pred_scores.device

        detections = []  # List[Dict[str, Tensor]]

        for boxes, scores, anchors, image_shape in zip(
            bbox_regression, pred_scores, image_anchors, image_shapes
        ):
            boxes = self.box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_boxes = []
            image_scores = []
            image_labels = []

            raw_boxes = torch.cat((boxes, scores), dim=1)

            image_box_ids = []

            for label in range(1, num_classes):
                score = scores[:, label]

                keep_idxs = score > self.score_thresh

                # ids of original boxes for this class > thresh
                thresh_ids = keep_idxs.clone().nonzero().squeeze(1).to(torch.long)
                assert torch.all(score[keep_idxs] == score[thresh_ids])

                score = score[keep_idxs]
                box = boxes[keep_idxs]

                # keep only topk scoring predictions
                num_topk = min(self.topk_candidates, score.size(0))
                score, idxs = score.topk(num_topk)
                box = box[idxs]

                final_ids = thresh_ids[idxs]
                image_box_ids.append(final_ids)

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(
                    torch.full_like(
                        score, fill_value=label, dtype=torch.int64, device=device
                    )
                )

            image_box_ids = torch.cat(image_box_ids, dim=0)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(
                image_boxes, image_scores, image_labels, self.nms_thresh
            )
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "raw_boxes": raw_boxes,
                    "box_ids": image_box_ids[keep],
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )
        return detections

    @staticmethod
    def _transform_postprocess(
        self,
        result: "List[Dict[str, Tensor]]",
        image_shapes: "List[Tuple[int, int]]",
        original_image_sizes: "List[Tuple[int, int]]",
    ) -> "List[Dict[str, Tensor]]":

        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(
            zip(result, image_shapes, original_image_sizes)
        ):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            # also for raw boxes
            if "raw_boxes" in pred:
                result[i]["raw_boxes"] = pred["raw_boxes"].clone()
                result[i]["raw_boxes"][:, :4] = resize_boxes(
                    result[i]["raw_boxes"][:, :4], im_s, o_im_s
                )

            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]["keypoints"] = keypoints
        return result

    @staticmethod
    def _postprocess_detections(
        self,
        head_outputs: "Dict[str, Tensor]",
        image_anchors: "List[Tensor]",
        image_shapes: "List[Tuple[int, int]]",
    ) -> "List[Dict[str, Tensor]]":

        bbox_regression = head_outputs["bbox_regression"]
        pred_scores = F.softmax(head_outputs["cls_logits"], dim=-1)

        num_classes = pred_scores.size(-1)
        device = pred_scores.device

        detections = []  # List[Dict[str, Tensor]]

        for boxes, scores, anchors, image_shape in zip(
            bbox_regression, pred_scores, image_anchors, image_shapes
        ):
            boxes = self.box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_boxes = []
            image_scores = []
            image_labels = []

            raw_boxes = torch.cat((boxes, scores), dim=1)

            image_box_ids = []

            for label in range(1, num_classes):
                score = scores[:, label]

                keep_idxs = score > self.score_thresh

                # ids of original boxes for this class > thresh
                thresh_ids = keep_idxs.clone().nonzero().squeeze(1).to(torch.long)
                assert torch.all(score[keep_idxs] == score[thresh_ids])

                score = score[keep_idxs]
                box = boxes[keep_idxs]

                # keep only topk scoring predictions
                num_topk = min(self.topk_candidates, score.size(0))
                score, idxs = score.topk(num_topk)
                box = box[idxs]

                final_ids = thresh_ids[idxs]
                image_box_ids.append(final_ids)

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(
                    torch.full_like(
                        score, fill_value=label, dtype=torch.int64, device=device
                    )
                )

            image_box_ids = torch.cat(image_box_ids, dim=0)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(
                image_boxes, image_scores, image_labels, self.nms_thresh
            )
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "raw_boxes": raw_boxes,
                    "box_ids": image_box_ids[keep],
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )
        return detections
