import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
from captum.attr import visualization as captumvis
import os
from .convert import tensorimg_to_npimg

plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['savefig.dpi'] = 300



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


def visualize_attribution(dataset, attribution, meta, figfile=None, show=True):

    if torch.is_tensor(attribution):
        attr_vis = np.transpose(
            attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)
        )
    else:
        attr_vis = attribution.squeeze()

    if np.allclose(attr_vis, 0):
        attr_vis += 1

    img_id, box_id, box_attr = meta["img_id"], meta["box_id"], meta["box_attr"]
    box_num, box, score, class_label = meta["box_num"], meta["box"], meta["score"], meta["label"]

    img_orig = dataset[img_id][0]

    img_orig_vis = tensorimg_to_npimg(img_orig)  # uint8

    idx_to_class = compute_idx_to_class(dataset.coco)

    if meta['explainer_engine'] == 'XGrad-CAM' or meta['explainer_engine'] == 'Grad-CAM++':
        cmap = 'jet'
        sign = 'positive'
    else:
        cmap = None
        sign = 'all'

    img_det_vis = draw_img_boxes(
        img_orig_vis,
        idx_to_class,
        pred={
            "boxes": [box],
            "scores": [score],
            "labels": [class_label],
            "box_nums": [box_num],
        },
    )

    fig, ax = captumvis.visualize_image_attr_multiple(
        attr_vis,
        img_det_vis,
        ["original_image", "blended_heat_map"],
        signs=[sign,sign],
        show_colorbar=True,
        alpha_overlay=0.5,
        fig_size=(8, 8),
        titles=[
            f"[{box_id}]({box_num}){idx_to_class[class_label]}: {score}",
            f"{meta['explainer_engine']}(box_attr={box_attr})",
        ],
        outlier_perc=1,
        use_pyplot=False,
        cmap=cmap
    )
    
    if figfile is not None:
    # figname = f"kshap_{img_id}_{box_id}_{box_attr}.png"
        visdir = os.path.dirname(figfile)
        os.makedirs(visdir, exist_ok=True)
        fig.savefig(figfile, dpi=300)

    if show:
        show_figure(fig)


def show_figure(fig):

    # create a dummy figure and use its
    # manager to display "fig"  
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    plt.show()
