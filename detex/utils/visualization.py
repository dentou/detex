import numpy as np
import cv2
from matplotlib import pyplot as plt



def draw_img_boxes(img, idx_to_class, gt=None, pred=None):
    """
    Data visualizer on the original image. Support both GT box input and proposal input.
    
    Input:
    - img: PIL Image input
    - idx_to_class: Mapping from the index (0-19) to the class name
    - bbox: GT bbox (in red, optional), a tensor of shape Nx5, where N is
            the number of GT boxes, 5 indicates (x_tl, y_tl, x_br, y_br, class)
    - pred: Predicted bbox (in green, optional), a tensor of shape N'x6, where
            N' is the number of predicted boxes, 6 indicates
            (x_tl, y_tl, x_br, y_br, class, object confidence score)
    """

    # img_copy = np.array(img).astype('uint8')
    img_copy = np.array(img).copy()
    # original_type = img_copy.dtype
    # if isinstance(img_copy, np.floating):
    #     img_copy = (255*img_copy).astype(np.uint8)

    text_font = cv2.FONT_HERSHEY_PLAIN
    text_thickness = 1

    

    if gt is not None:
        boxes = gt["boxes"].astype(np.int64)
        labels = gt.get("labels", None)

        for box_idx in range(boxes.shape[0]):
            one_bbox = boxes[box_idx]
            cv2.rectangle(img_copy, (one_bbox[0], one_bbox[1]), (one_bbox[2],
                        one_bbox[3]), (255, 0, 0), 2)
            if labels is not None:
                obj_cls = idx_to_class[labels[box_idx]]
                cv2.putText(img_copy, '%s' % (obj_cls),
                          (one_bbox[0], one_bbox[1]+15),
                          text_font, 1.0, (0, 0, 255), thickness=text_thickness)

    if pred is not None:
        boxes = np.array(pred["boxes"]).astype(np.int64)
        labels = pred.get("labels", None)
        scores = pred.get("scores", None)
        box_ids = pred.get("box_ids", None)
        box_nums = pred.get("box_nums", None)

        for box_idx in range(boxes.shape[0]):
            one_bbox = boxes[box_idx]
            cv2.rectangle(img_copy, (one_bbox[0], one_bbox[1]), (one_bbox[2],
                        one_bbox[3]), (0, 255, 0), 2)

            if box_nums is not None:
                box_num = box_nums[box_idx]
                text = f"({box_num})"
            else:
                text = f"({box_idx})"
            if labels is not None:
                obj_cls = idx_to_class[labels[box_idx]]
                text += f"{obj_cls}"
            
            if scores is not None:
                conf_score = scores[box_idx]
                text += f":{conf_score:.2f}"

            # if box_ids is not None:
            #     text = f"[{box_ids[box_idx]}]" + text

            cv2.putText(img_copy, text,
                        (one_bbox[0], one_bbox[1]+15),
                        text_font, 1.0, (0, 0, 255), thickness=text_thickness)

            
            # if labels is not None and scores is not None:
            #     obj_cls = idx_to_class[labels[box_idx]]
            #     conf_score = scores[box_idx]
            #     cv2.putText(img_copy, '%s, %.2f' % (obj_cls, conf_score),
            #                 (one_bbox[0], one_bbox[1]+15),
            #                 cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

    
    return img_copy


def compute_idx_to_class(coco):
    return {c["id"]:c["name"] for c in coco.loadCats(coco.getCatIds())}


def show_imgs(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]

    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(10, 10*len(imgs)))
    for i, img in enumerate(imgs):
        # img = img.detach()
        # img = TF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
