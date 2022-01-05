from re import S
import context
from detex.utils import (
    load_attribution,
    collect_metas,
    collapse_exp,
)
import argparse
import h5py

import numpy as np

import os

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
    parser.add_argument(
        "--baseline-value",
        default=0.5,
        type=float,
        help="Value assigned to perturbed pixels (in the range [0, 1])",
    )

    args = parser.parse_args()

    print(args)

    assert os.path.isfile(args.attribution_file), f"Cannot find file: {args.attribution_file}"
    filepath = os.path.abspath(args.attribution_file)
    baseline_value = args.baseline_value
    
   
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
        engine = meta["explainer_engine"]
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        attribution = collapse_exp(attribution.squeeze()) # (H, W, C)
        if engine.lower() == "kshap":
            num_pos = float(np.sum(attribution > 0))
            num_pos_in_box = float(np.sum(attribution[y1:y2, x1:x2] > baseline_value))
        elif engine.lower() == "xgrad-cam" or engine.lower() == "grad-cam++":
            num_pos = float(np.sum((attribution / 255.0) > baseline_value))
            num_pos_in_box = float(np.sum((attribution[x1:x2, y1:y2]/ 255.0) > baseline_value))

        box_area = float((x2-x1)*(y2-y1))

        results["num_pos"].append(num_pos)
        results["num_pos_in_box"].append(num_pos_in_box)
        results["box_area"].append(box_area)

    
    if engine == "XGrad-CAM" or engine == "Grad-CAM++":
        result_dir = f"data/results/cam/{engine.lower()}/pos_attr_in_box_{filename}"
    else:
        result_dir = f"data/results/{engine.lower()}/pos_attr_in_box_{filename}"
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
