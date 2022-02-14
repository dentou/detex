import warnings
import context
import os
import torchvision
import torch

from captum.attr import visualization as captumvis

from detex.explainers.cam.gradcams import XGradCAM, GradCAMPlusPlus
import numpy as np
import argparse

from detex.utils import (
    draw_img_boxes,
    compute_idx_to_class,
)
from detex.utils.convert import tensorimg_to_npimg

from tqdm.auto import tqdm

import seaborn as sns
from PIL import Image
from matplotlib import cm

from detex.utils.storage import save_attribution

sns.reset_orig()
sns.set(rc={"savefig.bbox": "tight", "figure.dpi": 300, "savefig.dpi": 300})


class GradCAMExplainer:
    def __init__(self, model, baseline, num_classes, explainer_engine, device):
        if 'cuda' in device.type:
            self.use_cuda = True
        else:
            self.use_cuda = False
        self.model = model
        self.model.score_thresh = baseline
        self.explainer_engine = explainer_engine
        if explainer_engine == 'XGrad-CAM':
            self.explainer = XGradCAM(self.model, baseline, num_classes, self.use_cuda)
        elif explainer_engine == 'Grad-CAM++':
            self.explainer = GradCAMPlusPlus(self.model, baseline, num_classes, self.use_cuda)

    def explain_single(self, img, label, bbox):

        assert len(img.shape) == 3, img.shape  # (C, H, W)
        assert torch.is_tensor(img)
        assert img.shape[0] == 3
        assert img.dtype == torch.float, img.dtype
        assert torch.is_tensor(bbox)
        assert len(bbox) == 4 # (xmin, ymin, xmax, ymax)
        assert (bbox[0] < bbox[2] and bbox[1] < bbox[3]) # xmin < xmax, ymin < ymax

        cam_cls, cam_box, output, index, overall_box_id = self.explainer(img, False, bbox, label)

        cmap = cm.get_cmap('jet')
        cam = Image.fromarray((cam_cls + cam_box) / 2 )
        img1 = Image.fromarray(tensorimg_to_npimg(img))
        overlay = cam.resize(img1.size, resample=Image.BILINEAR)
        overlay = np.asarray(overlay)
        attribution = (255 * cmap(overlay ** 2)[:, :, :3]).astype(np.uint8)

        return attribution, output, index, overall_box_id

    def explain_coco(self, dataset, img_id_list, filepath=None, visdir=None):
        for img_id in tqdm(img_id_list, desc="Picking img with id: "):

            model.zero_grad()
            img_orig = dataset[img_id][0]  # (C, H, W)

            img = img_orig.clone()

            with torch.no_grad():
                dets = self.model(img.unsqueeze(0).to(device))
                torch.cuda.empty_cache()

            if not dets:
                warnings.warn(
                    f"Empty detection for image {img_id}. Explanation skipped."
                )
                continue

            total_box_nums = len(dets[0]["boxes"])
            for box_num in tqdm(range(total_box_nums), desc="Explaning box_num: "):
                box = dets[0]["boxes"][box_num]
                score = dets[0]['scores'][box_num]
                index = dets[0]['labels'][box_num]

                attribution, _, index, overall_box_id = self.explain_single(
                    img,
                    label=index,
                    bbox=box
                )  # (H, W, C)
                

                if filepath:

                    attribution_save = attribution[None]  # (1, H, W, C)

                    if torch.is_tensor(index):
                        index = int(index.detach().numpy())
                    if torch.is_tensor(box):
                        box = list(box.detach().numpy())
                    if torch.is_tensor(score):
                        score = float(score)

                    meta = {
                        "explainer_engine": self.explainer_engine,
                        "img_id": img_id,
                        "box_id": overall_box_id,
                        "box_attr": index + 4,
                        "box_num": box_num,
                        "box": box,
                        "label": index,
                        "score": score,
                    }
                    save_attribution(attribution_save, filepath, meta)

                if visdir:

                    if torch.is_tensor(attribution):
                        attr_vis = np.transpose(
                            attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)
                        )
                    else:
                        attr_vis = attribution.squeeze()

                    img_orig_vis = tensorimg_to_npimg(img_orig)  # uint8

                    idx_to_class = compute_idx_to_class(dataset.coco)

                    img_det_vis = draw_img_boxes(
                        img_orig_vis,
                        idx_to_class,
                        pred={
                            "boxes": [box],
                            "scores": [score],
                            "labels": [index],
                            "box_nums": [box_num],
                        },
                    )

                    fig, ax = captumvis.visualize_image_attr_multiple(
                        attr_vis,
                        img_det_vis,
                        ["original_image", "blended_heat_map"],
                        ["all", "all"],
                        show_colorbar=True,
                        alpha_overlay=0.5,
                        fig_size=(8, 8),
                        titles=[
                            f"[{overall_box_id}]({box_num}){idx_to_class[index]}: {score}",
                            f"{self.explainer_engine}(box_attr={index + 4})",
                        ],
                        outlier_perc=1,
                        use_pyplot=False,
                    )
                    if self.explainer_engine == "XGrad-CAM":
                        figname = f"xgradcam_{img_id}_{overall_box_id}_{index + 4}.png"
                    elif self.explainer_engine == "Grad-CAM++":
                        figname = f"gradcampp_{img_id}_{overall_box_id}_{index + 4}.png"
                    os.makedirs(visdir, exist_ok=True)
                    figpath = os.path.join(visdir, figname)
                    print(f"Saving image to: {figpath}")
                    fig.savefig(figpath, dpi=300)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GradCAM")
    parser.add_argument(
        "--explainer-engine",
        default='XGrad-CAM',
        type=str,
        help="Explain with XGrad-CAM or Grad-CAM++",
    )

    parser.add_argument(
        "--first-images",
        type=int,
        default=1,
        help="Run XGrad-CAM/Grad-CAM++ on first x images in the dataset",
    )

    parser.add_argument(
        "--baseline-value",
        default=0.5,
        type=float,
        help="Value assigned to perturbed pixels (in the range [0, 1])",
    )

    parser.add_argument(
        "--result-file",
        default=None,
        type=str,
        help="HDF5 file to save attributions",
    )

    parser.add_argument(
        "--show-dir",
        default=None,
        type=str,
        help="Directory to store visualization",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Visualize and save in dir specified by --show-dir",
    )

    args = parser.parse_args()

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

    NUM_CLASSES = 91

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)

    model.to(device)

    img_id_list = np.arange(0, min(args.first_images, len(val_set))).tolist()

    if args.show:
        VISDIR = args.show_dir
    else:
        VISDIR = None

    CAM_FILE = args.result_file

    gradcam = GradCAMExplainer(
        model=model,
        baseline=args.baseline_value,
        num_classes=NUM_CLASSES,
        explainer_engine=args.explainer_engine,
        device=device
    )

    gradcam.explain_coco(
        val_set,
        img_id_list,
        visdir=VISDIR,
        filepath=CAM_FILE,
    )
