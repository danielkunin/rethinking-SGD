import glob
import os
import deepdish as dd
import numpy as np
import torch
from tqdm import tqdm
from utils import load
from utils import flags


def main():
    exp_path = f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}"
    step_names = glob.glob(f"{exp_path}/ckpt/*.tar")
    step_list = [int(s.split(".tar")[0].split("step")[1]) for s in step_names]
    device = load.device(ARGS.gpu)

    save_path = f"{exp_path}/feats"
    try:
        os.makedirs(save_path)
    except FileExistsError:
        if not ARGS.overwrite:
            print(
                "Feature directory exists and no-overwrite specified. Rerun with --overwrite"
            )
            quit()

    for in_filename, step in tqdm(
        sorted(list(zip(step_names, step_list)), key=lambda x: x[1])
    ):
        out_filename = f"{save_path}/step{step}.h5"

        if os.path.isfile(out_filename) and not ARGS.overwrite:
            print(f"\t{out_filename} already exists, skipping")
            continue

        checkpoint = torch.load(in_filename, map_location=device)

        # Metrics
        metrics = {}
        for m in ["train_loss", "test_loss", "accuracy1", "accuracy5"]:
            if m in checkpoint.keys():
                metrics[m] = np.array([checkpoint[m]])

        # Positions
        positions = []
        for name, tensor in checkpoint["model_state_dict"].items():
            positions.append(tensor.cpu().numpy())

        # Velocities
        velocities = []
        for group in checkpoint["optimizer_state_dict"]["param_groups"]:
            for p in group["params"]:
                param_state = checkpoint["optimizer_state_dict"]["state"][p]
                if "momentum_buffer" in param_state:
                    buf = param_state["momentum_buffer"]
                else:
                    buf = torch.zeros_like(p)
                velocities.append(buf.cpu().numpy())

        for p,v in zip(positions, velocities):
            assert p.shape == v.shape
        positions = np.concatenate([p.reshape(-1) for p in positions])
        velocities = np.concatenate([v.reshape(-1) for v in velocities])

        dd.io.save(
            out_filename, {
                "metrics": metrics,
                "position": positions,
                "velocity": velocities,
            }
        )


if __name__ == "__main__":
    parser = flags.extract()
    ARGS = parser.parse_args()
    main()
