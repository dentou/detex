from re import S
from captum import attr
from torch.utils import data
import context
from detex.utils import (
    load_attribution,
    collect_metas,
    tensorimg_to_npimg,
    compute_idx_to_class,
    draw_img_boxes,
    collapse_exp,
)
import argparse
from detex.models import SSDWrapper
import h5py

import torch
import numpy as np
import torchvision

import os
from matplotlib import pyplot as plt

from detex.utils.visualization import show_imgs, visualize_attribution
import torchvision.transforms.functional as TF
import torch.utils.data as TUD
from tqdm.auto import tqdm
from pathlib import Path


class PixelFlipper(torch.utils.data.Dataset):
    def __init__(
        self, img, attribution, flip_val, ratios=np.linspace(0, 1, 11).tolist()
    ) -> None:
        super().__init__()
        assert isinstance(img, np.ndarray)
        assert isinstance(attribution, np.ndarray)
        assert img.ndim == 3
        self.img = img
        self.attribution = collapse_exp(attribution.squeeze())
        self.total_num_pixels = np.prod(attribution.shape)

        self.flip_val = flip_val
        self.ratios = ratios

        self._compute_indices()

    def _compute_indices(self):
        self.top_ids = np.unravel_index(
            np.argsort(self.attribution, axis=None)[::-1], self.attribution.shape
        )[:2]

    def __len__(self):
        return len(self.ratios)

    def __getitem__(self, idx):

        img = self.get_numpy(idx)

        return TF.to_tensor(img)

    def get_numpy(self, idx):
        img = self.img.copy()

        ratio = self.ratios[idx]
        num_pixels = int(ratio * self.total_num_pixels)
        mask = np.zeros_like(img, dtype=bool)
        flipped_ids = tuple(t[:num_pixels] for t in self.top_ids)
        mask[flipped_ids] = True

        img[mask] = self.flip_val

        return img


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pixel Flipping")
    parser.add_argument(
        "attribution_file",
        type=str,
        help="Path to hdf5 attribution file",
    )
    parser.add_argument(
        "--ratio-points",
        nargs="?",
        type=int,
        default=51,
        help="NUmber of points between 0 and 1 (inclusive) used for getting flipping ratios",
    )

    parser.add_argument(
        "--flip-val",
        nargs="?",
        type=float,
        default=0.5,
        help="Value assigned to flipped pixels",
    )

    parser.add_argument(
        "--batch-size",
        nargs="?",
        default=16,
        type=int,
        help="Batch size for data loader",
    )

    parser.add_argument(
        "--num-workers",
        nargs="?",
        default=2,
        type=int,
        help="Number of workers for data loader",
    )

    parser.add_argument(
        "--show-legend",
        action="store_true",
        help="Show legend in plot",
    )



    args = parser.parse_args()

    print(args)

    assert os.path.isfile(args.attribution_file), f"Cannot find file: {args.attribution_file}"
    filepath = os.path.abspath(args.attribution_file)
    
   

    # filepath = "data/results/kshap/kshap_2000s_100i_colab.hdf5"

    filename = Path(filepath).resolve().stem

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

    metas = []
    with h5py.File(filepath, "r") as h:
        metas = collect_metas(h)

    idx_to_class = compute_idx_to_class(val_set.coco)

    allscores = []

    for meta in tqdm(metas, desc="Picking meta:"):
        attribution, meta = load_attribution(filepath, meta)

        img_id, box_id, box_attr = meta["img_id"], meta["box_id"], meta["box_attr"]

        img_orig = val_set[img_id][0]

        img = np.transpose(img_orig.detach().cpu().numpy().squeeze(), (1, 2, 0))

        forward_func = model.make_blackbox("captum", box_id, box_attr, device)

        ratios = np.linspace(0, 1, args.ratio_points).tolist()

        flipper = PixelFlipper(img, attribution, flip_val=args.flip_val, ratios=ratios)

        flipper_loader = TUD.DataLoader(
            flipper, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False
        )

        attrscores = []
        with torch.no_grad():
            for batch in tqdm(flipper_loader, desc="Flipping: "):
                scores = forward_func(batch)
                attrscores.extend(scores.tolist())

        allscores.append(attrscores)

    allscores = np.array(allscores)
    print(allscores)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    if box_attr >= 4:
        attrname = idx_to_class[box_attr - 4]
    else:
        attrname = ["x1", "y1", "x2", "y2"][box_attr]

    ax = axs[0]
    ax.plot(ratios, allscores.sum(axis=0))
    ax.set_xlabel("dropped ratio")
    ax.set_ylabel(f"sum of scores")
    ax.grid()

    ax = axs[1]
    ax.set_xlabel("dropped ratio")
    ax.set_ylabel(f"scores")
    ax.grid()
    for i, s in enumerate(allscores):
        meta = metas[i]
        img_id, box_id, box_attr = meta["img_id"], meta["box_id"], meta["box_attr"]
        ax.plot(ratios, s, label=f"{img_id}/{box_id}/{box_attr}")

    if args.show_legend:
        ax.legend()

    pixel_flip_dir = f"data/results/kshap/pixel_flip_{filename}"
    os.makedirs(pixel_flip_dir, exist_ok=True)

    scorefile = os.path.join(pixel_flip_dir, f"allscore.npy")
    np.save(scorefile, allscores)

    figfile = os.path.join(pixel_flip_dir, f"score_plot.png")
    fig.savefig(figfile, dpi=300)
