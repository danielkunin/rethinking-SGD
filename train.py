import json
import os
import shutil
import numpy as np
import deepdish as dd
import torch
import torch.nn as nn
from utils import load
from utils import optimize
from utils import flags
from utils import spectral


def main(ARGS):
    if ARGS.tpu:
        print_fn = xm.master_print
    else:
        print_fn = print

    ## Construct Result Directory ##
    if ARGS.expid == "":
        print_fn("WARNING: this experiment is not being saved.")
        setattr(ARGS, "save", False)
        save_path = None
    else:
        setattr(ARGS, "save", True)
        save_path = f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}"
        try:
            os.makedirs(save_path)
            os.makedirs(f"{save_path}/ckpt")
            os.makedirs(f"{save_path}/metrics")
        except FileExistsError:
            if not ARGS.overwrite:
                print_fn(
                    "Feature directory exists and no-overwrite specified. Rerun with --overwrite"
                )
                quit()
            shutil.rmtree(save_path)
            os.makedirs(save_path)
            os.makedirs(f"{save_path}/ckpt")
            os.makedirs(f"{save_path}/metrics")

    ## Save Args ##
    if ARGS.save:
        filename = save_path + "/hyperparameters.json"
        with open(filename, "w") as f:
            json.dump(ARGS.__dict__, f, sort_keys=True, indent=4)
        if ARGS.tpu:
            if xm.get_ordinal() == 0 and filename[0:5] == "gs://":
                from utils.gcloud import post_file_to_bucket

                post_file_to_bucket(filename)

    ## Random Seed and Device ##
    torch.manual_seed(ARGS.seed)
    device = load.device(ARGS.gpu, tpu=ARGS.tpu)

    ## Data ##
    print_fn("Loading {} dataset.".format(ARGS.dataset))
    input_shape, num_classes = load.dimension(ARGS.dataset)
    train_loader = load.dataloader(
        dataset=ARGS.dataset,
        batch_size=ARGS.train_batch_size,
        train=True,
        workers=ARGS.workers,
        datadir=ARGS.data_dir,
        tpu=ARGS.tpu,
    )
    test_loader = load.dataloader(
        dataset=ARGS.dataset,
        batch_size=ARGS.test_batch_size,
        train=False,
        workers=ARGS.workers,
        datadir=ARGS.data_dir,
        tpu=ARGS.tpu,
    )

    ## Model, Loss, Optimizer ##
    print_fn("Creating {}-{} model.".format(ARGS.model_class, ARGS.model))
    model = load.model(ARGS.model, ARGS.model_class)(
        input_shape=input_shape, num_classes=num_classes, pretrained=ARGS.pretrained,
        model_dir=ARGS.model_dir,
    )
    if len(ARGS.gpu.split(",")) > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    if ARGS.restore_path is not None:
        print_fn("Restoring model weights from {}".format(ARGS.restore_path))
        pretrained_dict = torch.load(ARGS.restore_path)
        model.load_state_dict(pretrained_dict["model_state_dict"])

    train_kwargs = {
        "batch_size": train_loader.batch_size,
        "dataset_size": len(train_loader.dataset),
        "num_batches": len(train_loader),
        "eval_mid_epoch": ARGS.lean_eval_mid_epoch,
        "spectral_path": ARGS.spectral_path,
    }
    if ARGS.tpu:
        train_kwargs.update(
            {"xrt_world_size": xm.xrt_world_size(), "xm_ordinal": xm.get_ordinal(),}
        )

    loss = load.loss(ARGS.loss)
    opt_class, opt_kwargs = load.optimizer(
        ARGS.optimizer, ARGS.momentum, ARGS.dampening, ARGS.nesterov,
    )
    opt_kwargs.update({"lr": ARGS.lr, "weight_decay": ARGS.wd})
    optimizer = opt_class(model.parameters(), **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=ARGS.lr_drops, gamma=ARGS.lr_drop_rate
    )

    ## Train ##
    print_fn("Training for {} epochs.".format(ARGS.epochs))
    optimize.train_eval_loop(
        model,
        loss,
        optimizer,
        scheduler,
        train_loader,
        test_loader,
        device,
        ARGS.epochs,
        ARGS.verbose,
        ARGS.save,
        save_freq=ARGS.save_freq,
        save_begin_epoch=ARGS.save_begin_epoch,
        save_path=save_path,
        lean_ckpt=ARGS.lean_ckpt,
        **train_kwargs,
    )

    # Final metrics save
    if ARGS.eigenvector or ARGS.hessian or ARGS.lanczos or ARGS.gradient:
        spectral_metrics = {}
        eigen_save_path = f"{save_path}/metrics"
        eigen_data_loader = load.dataloader(
            dataset=ARGS.dataset,
            batch_size=ARGS.eigen_batch_size,
            train=True,
            workers=ARGS.workers,
            datadir=ARGS.data_dir,
            tpu=ARGS.tpu,
            length=ARGS.eigen_data_length,
            shuffle=False,
            data_augment=True,
        )
        if ARGS.eigenvector:
            print("Computing Eigenvector")
            V, Lamb = spectral.subspace(
                loss,
                model,
                device,
                eigen_data_loader,
                ARGS.eigen_dims,
                ARGS.power_iters,
                eigen_save_path,
            )
            spectral_metrics["eigenvector"] = V
            spectral_metrics["eigenvalues"] = Lamb

        if ARGS.lanczos:
            print("Computing Eigenvectors with Lanczos")
            Lamb, V = spectral.get_hessian_eigenvalues(
                loss,
                model,
                device,
                eigen_data_loader,
                ARGS.eigen_dims,
            )
            spectral_metrics["eigenvector"] = V.cpu().numpy()
            spectral_metrics["eigenvalues"] = Lamb.cpu().numpy()

        if ARGS.hessian:
            print("Computing Hessian")
            H = spectral.hessian(loss, model, device, eigen_data_loader)
            spectral_metrics["hessian"] = H

        if ARGS.gradient:
            print("Computing Gradient")
            g = spectral.gradient(loss, model, device, eigen_data_loader)
            spectral_metrics["gradient"] = g

        dd.io.save(f"{eigen_save_path}/spectral.h5", spectral_metrics)


if __name__ == "__main__":
    parser = flags.train()
    ARGS = parser.parse_args()
    if ARGS.tpu:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp

        load.configure_tpu(ARGS.tpu)

        def _mp_fn(rank, args):
            main(args)

        xmp.spawn(_mp_fn, args=(ARGS,), nprocs=None, start_method="fork")
    else:
        main(ARGS)
