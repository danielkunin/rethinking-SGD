import os
from tqdm import tqdm
import metrics.helper as utils
import numpy as np

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


metric_fns = {
    "performance": performance,
    "hessian_eigenprojection": hessian_eigenprojection,
}
