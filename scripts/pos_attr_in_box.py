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

import json


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Percent inside box")
    parser.add_argument(
        "attribution_file",
        type=str,
        help="Path to hdf5 attribution file",
    )

    args = parser.parse_args()

    print(args)

    assert os.path.isfile(args.attribution_file), f"Cannot find file: {args.attribution_file}"
    filepath = os.path.abspath(args.attribution_file)
    
   
    filename = Path(filepath).resolve().stem

    metas = []
    with h5py.File(filepath, "r") as h:
        metas = collect_metas(h)


    results = {
        "num_pos": [],
        "num_pos_in_box": [],
        "box_area": [],
    }


    for meta in tqdm(metas, desc="Picking meta:"): 
        attribution, meta = load_attribution(filepath, meta)

        img_id, box_id, box_attr = meta["img_id"], meta["box_id"], meta["box_attr"]

        box = meta["box"]
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        attribution = collapse_exp(attribution.squeeze()) # (H, W, C)
        num_pos = float(np.sum(attribution > 0))
        num_pos_in_box = float(np.sum(attribution[y1:y2, x1:x2] > 0))

        box_area = float((x2-x1)*(y2-y1))

        results["num_pos"].append(num_pos)
        results["num_pos_in_box"].append(num_pos_in_box)
        results["box_area"].append(box_area)

    

    result_dir = f"data/results/kshap/pos_attr_in_box_{filename}"
    os.makedirs(result_dir, exist_ok=True)

    total_num_pos = sum(results["num_pos"])
    total_num_pos_in_box = sum(results["num_pos_in_box"])
    total_box_area = sum(results["box_area"])

    print(f"Percentage: {100*total_num_pos_in_box/total_num_pos}%")
    print(f"Box coverage: {100*total_num_pos_in_box/total_box_area}%")
    print(f"IoU: {100*total_num_pos_in_box/(total_box_area + total_num_pos - total_num_pos_in_box)}%")

    resultfile = os.path.join(result_dir, f"results.json")
    with open(resultfile, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
