import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops
from torchvision.models.detection import _utils as det_utils
from typing import Dict, List, Tuple
from collections import OrderedDict


class SSD300VGG16FeatureExtractor:
    """Class for extracting activations and
    registering gradients from targetted intermediate layers"""

    def __init__(self, model, num_classes, bbox):
        self.model = model
        self.bbox = bbox
        self.transform = self.model._modules['transform']
        self.gradients = []
        self.num_columns = num_classes
        self.box_coder = det_utils.BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.score_thresh = 0.5
        self.nms_thresh = 0.4
        self.topk_candidates = 400
        self.detections_per_img = 200
    
    @staticmethod
    def _resize_boxes(boxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=boxes.device)
            / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]
        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = boxes.unbind(1)

        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)

    def _save_gradient(self, grad):
        self.gradients.append(grad)

    def _postprocess_detections(self, head_outputs: Dict[str, Tensor], image_anchors: List[Tensor],
                               image_shapes: List[Tuple[int, int]]) -> List[Dict[str, Tensor]]:
        bbox_regression = head_outputs['bbox_regression']
        pred_scores = F.softmax(head_outputs['cls_logits'], dim=-1)

        num_classes = pred_scores.size(-1)
        device = pred_scores.device

        detections: List[Dict[str, Tensor]] = []

        for boxes, scores, anchors, image_shape in zip(bbox_regression, pred_scores, image_anchors, image_shapes):
            boxes = self.box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_boxes = []
            image_scores = []
            image_labels = []
            for label in range(1, num_classes):
                score = scores[:, label]

                keep_idxs = score > self.score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]

                # keep only topk scoring predictions
                num_topk = min(self.topk_candidates, score.size(0))
                score, idxs = score.topk(num_topk)
                box = box[idxs]

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64, device=device))

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]

            detections.append({
                'boxes': image_boxes[keep],
                'scores': image_scores[keep],
                'labels': image_labels[keep],
            })
        return detections

    def find_target_layer(self, x):
        original_image_sizes = []
        for img in x:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        x, _ = self.transform(x)
        for name, module in self.model._modules.items():
            if name == 'backbone':
                features = module(x.tensors)
                if isinstance(features, torch.Tensor):
                    features = OrderedDict([('0', features)])
                features = list(features.values())
            elif name == 'head':
                head_outputs = {}
                for n, m in module._modules.items():
                    if n == 'classification_head':
                        for i, j in m._modules['module_list']._modules.items():
                            y = j(features[int(i)])
                            N, _, H, W = y.shape
                            y = y.view(N, -1, self.num_columns, H, W)
                            y = y.permute(0, 3, 4, 1, 2)
                            y = y.reshape(N, -1, self.num_columns)
                    else:
                        head_outputs['bbox_regression'] = []
                        for i, j in m._modules['module_list']._modules.items():
                            y = j(features[int(i)])
                            N, _, H, W = y.shape
                            y = y.view(N, -1, 4, H, W)
                            y = y.permute(0, 3, 4, 1, 2)
                            y = y.reshape(N, -1, 4)

                            head_outputs['bbox_regression'].append(y)
            elif name == 'anchor_generator':
                anchors = module(x, features)

        count = [0]
        for scale_idx in range(6):
            for box_id in range(len(head_outputs['bbox_regression'][scale_idx])):
                count.append(count[scale_idx] + head_outputs['bbox_regression'][scale_idx][box_id].size()[0])
                boxes = self.box_coder.decode_single(head_outputs['bbox_regression'][scale_idx][box_id], anchors[0][count[scale_idx]:count[scale_idx+1]])
                boxes = box_ops.clip_boxes_to_image(boxes, x.image_sizes[0])
                boxes = self._resize_boxes(boxes, x.image_sizes[0],original_image_sizes[0])
                for box in boxes:
                    if torch.allclose(box.detach(), self.bbox):
                        self.target_layer = scale_idx

    def __call__(self, input_imgs, zero_out=None):
        outputs = []
        self.gradients = []
        original_image_sizes = []
        for img in input_imgs:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        input_imgs, _ = self.transform(input_imgs)
        for name, module in self.model._modules.items():
            if name == 'backbone':
                features = OrderedDict()
                for name1, module1 in module._modules.items():
                    if name1 == 'features':
                        if self.target_layer == 0:
                            for name2, module2 in module1._modules.items():
                                if name2 == '22':
                                    y = module2(y)
                                    y.register_hook(self._save_gradient)
                                    outputs += [y]
                                    features[str(self.target_layer)] = y
                                else:
                                    if name2 == '0':
                                        y = module2(input_imgs.tensors)
                                    else:
                                        y = module2(y)
                        else:
                            y = module1(input_imgs.tensors)
                            features['0'] = y
                    else:
                        if self.target_layer > 0:
                            for name2, module2 in module1._modules.items():
                                if name2 == str(self.target_layer - 1):
                                    if name2 == '0':
                                        for name3, module3 in module2._modules.items():
                                            if name3 == '7':
                                                for name4, module4 in module3._modules.items():
                                                    if name4 == '4':
                                                        y = module4(y)
                                                        y.register_hook(self._save_gradient)
                                                        outputs += [y]
                                                        features[str(self.target_layer)] = y
                                                    else:
                                                        y = module4(y)
                                            else:
                                                y = module3(y)
                                    else:
                                        for name3, module3 in module2._modules.items():
                                            if name3 == '3':
                                                y = module3(y)
                                                y.register_hook(self._save_gradient)
                                                outputs += [y]
                                                features[str(self.target_layer)] = y
                                            else:
                                                y = module3(y)
                                else:
                                    y = module2(y)
                                    features[str(int(name2) + 1)] = y
                        else:
                            y = module1(y)
                        
                features = list(features.values())

            elif name == 'head':
                head_outputs = {}
                raw_boxes_per_scale = {}
                for name1, module1 in module._modules.items():
                    if name1 == 'classification_head':
                        head_outputs['cls_logits'] = []
                        raw_boxes_per_scale['cls_logits'] = []
                        for name2, module2 in module1._modules['module_list']._modules.items():
                            y = module2(features[int(name2)])
                            N, _, H, W = y.shape
                            y = y.view(N, -1, self.num_columns, H, W)
                            y = y.permute(0, 3, 4, 1, 2)
                            y = y.reshape(N, -1, self.num_columns)

                            head_outputs['cls_logits'].append(y)
                            raw_boxes_per_scale['cls_logits'].append(y)
                    else:
                        head_outputs['bbox_regression'] = []
                        raw_boxes_per_scale['bbox_regression'] = []
                        for name2, module2 in module1._modules['module_list']._modules.items():
                            y1 = module2(features[int(name2)])
                            N, _, H, W = y1.shape
                            y1 = y1.view(N, -1, 4, H, W)
                            y1 = y1.permute(0, 3, 4, 1, 2)
                            y1 = y1.reshape(N, -1, 4)

                            head_outputs['bbox_regression'].append(y1)
                            raw_boxes_per_scale['bbox_regression'].append(y1)

            elif name == 'anchor_generator':
                anchors = module(input_imgs, features)
        
        head_outputs['cls_logits'] = torch.cat(head_outputs['cls_logits'], dim=1)
        head_outputs['bbox_regression'] = torch.cat(head_outputs['bbox_regression'], dim=1)
        detections = self._postprocess_detections(head_outputs, anchors, input_imgs.image_sizes)
        detections = self.transform.postprocess(detections, input_imgs.image_sizes, original_image_sizes)

        return input_imgs, outputs, detections, raw_boxes_per_scale, anchors

class ModelOutputs:
    """Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers."""

    def __init__(self, model, num_classes, bbox):
        self.model = model
        if 'SSD' in model._modules['backbone'].__class__.__name__:
            self.feature_extractor = SSD300VGG16FeatureExtractor(model, num_classes, bbox)

        else:
            raise ValueError("Model is not supported! Please input SSD model!")
          
    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, input_imgs, zero_out=False):
        # Find which layer the target bounding box is at
        self.feature_extractor.find_target_layer(input_imgs)
        transformed_input, target_activations, detections, raw_boxes_per_scale, anchors = self.feature_extractor(input_imgs, zero_out)

        return transformed_input, target_activations, detections, raw_boxes_per_scale, anchors