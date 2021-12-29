import warnings
from captum import attr
import context
import os
import torchvision
import torchvision.transforms.functional as TF
import torch

import captum
from captum.attr import visualization as captumvis

from detex.models import SSDWrapper
import numpy as np
from skimage import segmentation as skseg

from detex.utils import draw_img_boxes, compute_idx_to_class, set_seed, segment, collect_detections
from detex.utils.convert import tensorimg_to_npimg

from tqdm.auto import tqdm

@torch.no_grad()
def kernelshap(img, segment_mask, model, box_id, box_attr, seed):

    assert len(img.shape) == 3, img.shape  # (C, H, W)
    assert torch.is_tensor(img)
    assert img.shape[0] == 3
    assert img.dtype == torch.float, img.dtype

    assert len(segment_mask.shape) == 3, segment_mask.shape  # (1, H, W)

    f = model.make_blackbox("captum", box_id, box_attr, device)
    ks = captum.attr.KernelShap(f)

    set_seed(seed)
    feature_mask = segment_mask.unsqueeze(0)

    input_img = img.unsqueeze(0)

    attributions = ks.attribute(
        input_img,
        feature_mask=feature_mask,
        baselines=0.5,
        n_samples=1000,
        perturbations_per_eval=16,
        show_progress=True,
    ).cpu()


    return attributions



if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(".")
    DATA_DIR = os.path.join(ROOT_DIR, "data")

    print(DATA_DIR)

    VAL_IMG_DIR = os.path.join(DATA_DIR, "val2017")
    VAL_ANN_FILE = os.path.join(DATA_DIR, "annotations", "instances_val2017.json")

    val_set = torchvision.datasets.CocoDetection(
        root=VAL_IMG_DIR,
        annFile=VAL_ANN_FILE,
        transform=torchvision.transforms.ToTensor(),
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    model = SSDWrapper(torchvision.models.detection.ssd300_vgg16(pretrained=True))
    model.eval()

    model.to(device)

    img_id_list = [0, ]
    for img_id in tqdm(img_id_list, desc="Picking img with id: "):

        img_orig = val_set[img_id][0]  # (C, H, W)

        img = img_orig.clone()

        spixel_mask = segment(img)  # (H, W)
        segment_mask = TF.to_tensor(spixel_mask)  # (1, H, W)

        with torch.no_grad():
            orig_dets = model(img.unsqueeze(0).to(device))
            dets = collect_detections(orig_dets)
            del orig_dets
            torch.cuda.empty_cache()
        
        if not dets:
            warnings.warn(f"Empty detection for image {img_id}. Explanation skipped.")
            continue

        total_box_nums = len(dets[0]["box_ids"])
        for box_num in tqdm(range(total_box_nums), desc="Explaning box_num: "):
            box_id = dets[0]["box_ids"][box_num]  # only have 1 image
            class_label = dets[0]["labels"][box_num]
            box = dets[0]["boxes"][box_num]
            score = dets[0]["scores"][box_num]
            box_attr = 4 + class_label

            attribution = kernelshap(
                img,
                segment_mask=segment_mask,
                model=model,
                box_id=box_id,
                box_attr=box_attr,
                seed=42,
            ) # (1, C, H, W)

            
            if torch.is_tensor(attribution):
                attr_vis = np.transpose(attribution.squeeze().cpu().detach().numpy(), (1,2,0))
            else:
                attr_vis = attribution.squeeze()

            

            img_orig_vis = tensorimg_to_npimg(img_orig) # uint8


            idx_to_class = compute_idx_to_class(val_set.coco)

            img_det_vis = draw_img_boxes(
                img_orig_vis, 
                idx_to_class, 
                pred= {
                    "boxes": [box],
                    "scores": [score],
                    "labels": [class_label],
                    "box_nums": [box_num],
                })

            fig, ax = captumvis.visualize_image_attr_multiple(
                attr_vis,
                img_det_vis,
                ["original_image", "blended_heat_map"],
                ["all", "all"],
                show_colorbar=True,
                alpha_overlay=0.85,
                fig_size=(8, 8),
                titles=[f"[{box_id}]({box_num}){idx_to_class[class_label]}: {score}", "KSHAP"],
                outlier_perc=1,
                use_pyplot=False,
            )
            figname = f"kshap_{img_id}_{box_id}_{box_attr}.png"
            fig.savefig(os.path.join(DATA_DIR, figname), dpi=300)


