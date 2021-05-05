import os
from tqdm import tqdm
import metrics.helper as utils
import numpy as np
import glob
import torch

from metrics.hessian import hessian_eigenprojection

def performance(model, feats_dir, steps, **kwargs):
    metrics = {}
    for i in tqdm(range(len(steps))):
        step = steps[i]
        feats_path = f"{feats_dir}/step{step}.h5"
        if os.path.isfile(feats_path):
            feature_dict = utils.get_features(
                feats_path=feats_path,
                group="metrics",
                keys=["accuracy1", "accuracy5", "train_loss", "test_loss"],
            )
            metrics[step] = feature_dict
        metrics["steps"] = steps
    return {"performance": metrics}

def performance_from_ckpt(model, feats_dir, steps, **kwargs):
    ckpt_dir = feats_dir.replace("feats", "ckpt")
    step_names = glob.glob(
        f"{ckpt_dir}/*.tar"
    )
    steps = sorted(
        [int(s.split(".tar")[0].split("step")[1]) for s in step_names]
    )
    metric_keys = [
        "train_loss",
        "train_accuracy1",
        "train_accuracy5",
        "test_loss",
        "test_accuracy1",
        "test_accuracy5",
        "step",
    ]
    metrics = {m: [] for m in metric_keys}
    for i in tqdm(range(len(steps))):
        step = steps[i]
        ckpt = torch.load(f"{ckpt_dir}/step{step}.tar")
        for m in metric_keys:
            if m in ckpt.keys():
                metrics[m].append(ckpt[m])

    metrics = {k:np.array(v) for k,v in metrics.items()}
    return {"performance": metrics}


metric_fns = {
    "performance": performance,
    "performance_from_ckpt": performance_from_ckpt,
    "hessian_eigenprojection": hessian_eigenprojection,
}
