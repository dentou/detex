from torch.utils import data
import context
from detex.utils import (
    load_attribution,
    collect_metas,
    tensorimg_to_npimg,
    compute_idx_to_class,
    draw_img_boxes,
)
import h5py

import argparse

import torch
import numpy as np
import torchvision

import os

from detex.utils.visualization import visualize_attribution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attribution filepath")
    parser.add_argument(
        "--filepath",
        default=None,
        type=str,
        help="hdf5 attribution file path",
    )
    args = parser.parse_args()

    filepath = args.filepath

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

    metas = []
    with h5py.File(filepath, "r") as h:
        metas = collect_metas(h)


    for count, meta in enumerate(metas):
        attribution, meta = load_attribution(filepath, meta)
        img_id, box_id, box_attr = meta["img_id"], meta["box_id"], meta["box_attr"]
        if count > 20:
            break
        visualize_attribution(
            val_set,
            attribution,
            meta,
            show=False,
            figfile=f"docs/images/{meta['explainer_engine'].lower()}/{filepath.split('/')[-1].split('.')[0]}_{img_id}_{box_id}_{box_attr}.png",
        )
