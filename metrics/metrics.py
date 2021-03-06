import os
from tqdm import tqdm
import metrics.helper as utils
import numpy as np
import glob
import torch
import deepdish as dd

from metrics.hessian import hessian_eigenprojection
from metrics.hessian import fft
from metrics.hessian import load_weight_and_grad

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
        "train_batch_accuracy1",
        "train_batch_accuracy5",
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


def weight_norm(model, feats_dir, steps, **kwargs):
    norms = []
    for i in tqdm(range(0, len(steps))):
        step = steps[i]
        weight, _ = load_weight_and_grad(step, feats_dir)
        norms.append(np.linalg.norm(weight))
    norms = np.array(norms)
    return {"weight_norm": norms}

def loss_diff_from_ckpt(model, feats_dir, steps, **kwargs):
    ckpt_dir = feats_dir.replace("feats", "ckpt")
    step_names = glob.glob(
        f"{ckpt_dir}/*.tar"
    )
    steps = sorted(
        [int(s.split(".tar")[0].split("step")[1]) for s in step_names]
    )
    metric_keys = [
        "vel_norm",
        "pos_norm",
        "pos_dot",
        #"test_loss",
        #"test_accuracy1",
        #"test_accuracy5",
    ]
    metrics = {m: [] for m in metric_keys}
    for i in tqdm(range(len(steps))):
        step = steps[i]
        ckpt = torch.load(f"{ckpt_dir}/step{step}.tar")
        if "vel_norm" in ckpt.keys():
            for m in metric_keys:
                if type(ckpt[m]) == torch.Tensor:
                    ckpt[m] = ckpt[m].cpu().numpy()
                metrics[m].append(ckpt[m])
    metrics = {k:np.array(v) for k,v in metrics.items()}

    for k in metric_keys:
        if k is "vel_norm" or k is "pos_norm":
            #metrics[k] = metrics[k][::2]**2
            metrics[k] = metrics[k]**2
        elif k is "pos_dot":
            metrics[k] = metrics[k]
        else:
            metrics[k] = (metrics[k][1::2] - metrics[k][::2])**2
    return {"loss_diff": metrics}


def dist_from_start_from_ckpt(model, feats_dir, steps, **kwargs):
    ckpt_dir = feats_dir.replace("feats", "ckpt")
    step_names = glob.glob(
        f"{ckpt_dir}/*.tar"
    )
    steps = sorted(
        [int(s.split(".tar")[0].split("step")[1]) for s in step_names]
    )
    metric_keys = [
        "dist_from_start",
        #"projected_pos",
        #"projected_vel",
    ]
    metrics = {m: [] for m in metric_keys}
    for i in tqdm(range(len(steps))):
        step = steps[i]
        ckpt = torch.load(f"{ckpt_dir}/step{step}.tar")
        if "dist_from_start" in ckpt.keys():
            for m in metric_keys:
                if type(ckpt[m]) is list or type(ckpt[m]) is tuple:
                    ckpt[m] = ckpt[m][0]
                if type(ckpt[m]) is torch.Tensor:
                    ckpt[m] = ckpt[m].cpu().numpy()
                metrics[m].append(ckpt[m])
    metrics = {k:np.array(v) for k,v in metrics.items()}
    return {"dist_from_start": metrics}

def load_weight_and_grad(step, feats_dir):
    load_path = f"{feats_dir}/step{step}.h5"
    weight = dd.io.load(load_path, f"/position")
    grad = dd.io.load(load_path, f"/velocity")
    return weight, grad


def loss_diff(model, feats_dir, steps, **kwargs):
    ckpt_dir = feats_dir.replace("feats", "ckpt")
    metric_keys = [
        "test_loss",
        "test_accuracy1",
        "test_accuracy5",
    ]

    weights = []
    grads = []
    metrics = {m: [] for m in metric_keys}
    for i in tqdm(range(0, len(steps))):
        step = steps[i]
        weight, grad = load_weight_and_grad(step, feats_dir)
        weights.append(weight)
        grads.append(grad)

        # Load accuracies from checkpoints at every epoch
        ckpt = torch.load(f"{ckpt_dir}/step{step}.tar")
        for m in metric_keys:
            metrics[m].append(ckpt[m])
    metrics = {k:np.array(v) for k,v in metrics.items()}
    for k in metric_keys:
        metrics[k] = (metrics[k][1:] - metrics[k][:-1])**2

    weights = np.array(weights)
    grads = np.array(grads)

    metrics["weight_diff_norm"] = np.linalg.norm((weights[1:] - weights[:-1]), axis=1)**2

    return {"loss_diff": metrics}


metric_fns = {
    "performance": performance,
    "performance_from_ckpt": performance_from_ckpt,
    "loss_diff": loss_diff,
    "loss_diff_from_ckpt": loss_diff_from_ckpt,
    "dist_from_start_from_ckpt": dist_from_start_from_ckpt,
    "hessian_eigenprojection": hessian_eigenprojection,
    "fft": fft,
    "weight_norm": weight_norm,
}
