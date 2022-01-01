import h5py
import os
import numpy as np


def collapse_exp(exp):
    if (exp[..., 0] == exp[..., 1]).all() and (exp[..., 0] == exp[..., 2]).all():
        return exp[..., 0:1]
    else:
        return exp

def expand_exp(exp):
    if isinstance(exp, np.ndarray) and exp.shape[-1] == 1:
        exp = np.repeat(exp, 3, axis=-1)
    return exp


def save_attribution(attribution: np.ndarray, filepath: str, meta: dict):
    """Store attribution

    Args:
        attribution (np.ndarray): [description]
        filepath (str): path to hdf5 file
        meta (Dict): dictionary with keys:
            img_id (int): image id in dataset
            box_id (int): box id in detector's head
            box_attr (int): column number within raw_box (x1, y1, x2, y2, score_for_class0, ..., score_for_class(N-1))
                for example, box_attr=4+class_label gives score for respective class
            box_num: box number within image, for visualization
            box: numpy array (4, ), for visulization
            label: int, for visualization
            score: float, for visualization
    """
    img_id = meta["img_id"]
    box_id = meta["box_id"]
    box_attr = meta["box_attr"]
    box_num = meta["box_num"]
    box = meta["box"]
    label = meta["label"]
    score = meta["score"]

    filedir = os.path.dirname(filepath)
    os.makedirs(filedir, exist_ok=True)
    
    with h5py.File(filepath, 'a') as h:
        g_img_id = h.require_group(str(img_id))
        g_box_id = g_img_id.require_group(str(box_id))

        g_box_id.attrs["box_num"] = box_num
        g_box_id.attrs["box"] = box
        g_box_id.attrs["label"] = label 
        g_box_id.attrs["score"] = score


        attribution = collapse_exp(attribution)
        g_box_id.require_dataset(str(box_attr), data=attribution, shape=attribution.shape, dtype=attribution.dtype)



def allkeys(obj, leaves_only=False):
    "Recursively find all keys in an h5py.Group."
    keys = ()

    if not leaves_only or (leaves_only and isinstance(obj, h5py.Dataset)):
        keys += (obj.name,)

    if isinstance(obj, h5py.Group):
        for _, value in obj.items():
            keys += allkeys(value, leaves_only=leaves_only)

    return keys

def collect_metas(hfile):
    keys = allkeys(hfile, leaves_only=True)
    metas = []
    for k in keys:
        _, img_id, box_id, box_attr = k.split("/")
        img_id, box_id, box_attr = int(img_id), int(box_id), int(box_attr)
        metas.append({"img_id": img_id, "box_id": box_id, "box_attr": box_attr})

    return metas


def load_attribution(filepath, meta):
    """Load attribution

    Args:
        filepath (str): path to hdf5 file
        meta (Dict): dictionary with keys:
            img_id (int): image id in dataset
            box_id (int): box id in detector's head
            box_attr (int): column number within raw_box (x1, y1, x2, y2, score_for_class0, ..., score_for_class(N-1))
                for example, box_attr=4+class_label gives score for respective class
    Returns:
        attribution: stored numpy array
        meta (Dict): meta dict with additional keys:
            box_num: box number within image, for visualization
            box: numpy array (4, ), for visulization
            label: int, for visualization
            score: float, for visualization

    """
    img_id = meta["img_id"]
    box_id = meta["box_id"]
    box_attr = meta["box_attr"]

    with h5py.File(filepath, 'r') as h:
        g_img_num = h[str(img_id)]
        g_box_id = g_img_num[str(box_id)]

        for k, v in g_box_id.attrs.items():
            meta[k] = v

        attribution = np.array(g_box_id[str(box_attr)])
        attribution = expand_exp(attribution)
    return attribution, meta