import os
import json
import shutil
import copy
import torch
import torch.nn as nn
import deepdish as dd
import numpy as np
from utils import flags
from utils import load

def eval(model, loss, dataloader, device, train=True):
    print_fn = print
    if device.type == "xla":
        import torch_xla.core.xla_model as xm

        print_fn = xm.master_print

    if train:
      model.train()
    else:
      model.eval()

    total = 0
    correct1 = 0
    correct5 = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:, :1].sum().item()
            correct5 += correct[:, :5].sum().item()
            total_samples += data.size()[0]
    average_loss = 1.0 * total / total_samples
    accuracy1 = 100.0 * correct1 / total_samples
    accuracy5 = 100.0 * correct5 / total_samples

    if device.type == "xla":
        average_loss = xm.mesh_reduce("test_average_loss", average_loss, np.mean)
        accuracy1 = xm.mesh_reduce("test_accuracy1", accuracy1, np.mean)
        accuracy5 = xm.mesh_reduce("test_accuracy5", accuracy5, np.mean)
    return average_loss, accuracy1, accuracy5

def shift_and_eval(model, loss, train_loader, test_loader, device, shift,
                   verbose=False):
    print_fn = print
    if device.type == "xla":
        import torch_xla.core.xla_model as xm

        print_fn = xm.master_print

    with torch.no_grad():
        n = 0
        for p in model.parameters():
            k = p.numel()
            p.add_(shift[n:n+k].reshape(p.shape))
            n += k
    print_fn("Shifted the model")
    test_loss, test_acc, _ = eval(model, loss, test_loader, device,
        train=False)
    if verbose:
        print_fn(
            f"Test evaluation: Average Loss: {test_loss:.4f}, "
            f"Top 1 test Accuracy: {test_acc:.2f}%)"
        )

    train_loss, train_acc, _ = eval(model, loss, train_loader, device,
        train=False)
    if verbose:
        print_fn(
            f"Train evaluation: Average Loss: {train_loss:.4f}, "
            f"Top 1 train Accuracy: {train_acc:.2f}%)"
        )

    return train_loss, train_acc, test_loss, test_acc


def extend_parser(parser):
    parser.add_argument('--x-min', type=float, default=-0.1,
                    help='Min x limit in euclidean coordinates')
    parser.add_argument('--x-max', type=float, default=0.1,
                    help='Max y limit in euclidean coordinates')
    parser.add_argument('--y-min', type=float, default=-0.1,
                    help='Max x limit in euclidean coordinates')
    parser.add_argument('--y-max', type=float, default=0.1,
                    help='Min y limit in euclidean coordinates')
    parser.add_argument('--x-samples', type=int, default=40,
                    help='Number of points to sample along x')
    parser.add_argument('--y-samples', type=int, default=40,
                    help='Number of points to sample along y')
    parser.add_argument('--x-begin', type=int, default=0,
                    help='Index of the grid to begin with (inclusive) along x')
    parser.add_argument('--x-end', type=int, default=40,
                    help='Index of the grid to end with (exclusive) along x')
    parser.add_argument('--y-begin', type=int, default=0,
                    help='Index of the grid to begin with (inclusive) along x')
    parser.add_argument('--y-end', type=int, default=40,
                    help='Index of the grid to end with (exclusive) along x')
    parser.add_argument('--u-idx', type=int, default=0,
                    help='Index of the eigenvector along which to shift the '
                         'model in the x coordinate of the grid')
    parser.add_argument('--v-idx', type=int, default=1,
                    help='Index of the eigenvector along which to shift the '
                         'model in the y coordinate of the gird')
    parser.add_argument(
        "--spectral-path",
        type=str,
        default=None,
        help="Path to load eigenvalues and eigenvectors from.",
    )
    parser.add_argument('--data-length', type=int, default=50000,
                    help='Number of examples to subset from the dataset.')
    return parser


def main(ARGS):
    if ARGS.tpu:
        import torch_xla.core.xla_model as xm

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
        exp_path = f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}"
        save_path = f"{exp_path}/grid"
        try:
            os.makedirs(exp_path)
            os.makedirs(save_path)
        except FileExistsError:
            if not ARGS.overwrite:
                print_fn(
                    "Feature directory exists and no-overwrite specified. Rerun with --overwrite"
                )
                quit()

    filename = exp_path + "/hyperparameters.json"
    with open(filename, "w") as f:
        json.dump(ARGS.__dict__, f, sort_keys=True, indent=4)
        if ARGS.tpu:
            if xm.get_ordinal() == 0 and filename[0:5] == "gs://":
                from utils.gcloud import post_file_to_bucket

                post_file_to_bucket(filename)

    torch.manual_seed(0)
    device = load.device(ARGS.gpu, tpu=ARGS.tpu)

    print_fn("Loading {} dataset.".format(ARGS.dataset))
    input_shape, num_classes = load.dimension(ARGS.dataset)
    train_loader = load.dataloader(
        dataset=ARGS.dataset,
        batch_size=ARGS.train_batch_size,
        train=True,
        workers=ARGS.workers,
        datadir=ARGS.data_dir,
        tpu=ARGS.tpu,
        length=ARGS.data_length,
        shuffle=False,
        data_augment=False,
    )
    test_loader = load.dataloader(
        dataset=ARGS.dataset,
        batch_size=ARGS.test_batch_size,
        train=False,
        workers=ARGS.workers,
        datadir=ARGS.data_dir,
        tpu=ARGS.tpu,
        length=ARGS.data_length,
    )

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
    }
    if ARGS.tpu:
        train_kwargs.update(
            {"xrt_world_size": xm.xrt_world_size(), "xm_ordinal": xm.get_ordinal(),}
        )

    loss = load.loss("ce")

    # Grid eval
    m = sum(p.numel() for p in model.parameters())
    eigenvectors = dd.io.load(ARGS.spectral_path, "/eigenvector")
    u = torch.tensor(eigenvectors[:,ARGS.u_idx], device=device)
    v = torch.tensor(eigenvectors[:,ARGS.v_idx], device=device)
    position = []
    with torch.no_grad():
        for p in model.parameters():
            position.append(p.flatten())
        position = torch.cat(position)
        cu0 = torch.dot(position, u)
        cv0 = torch.dot(position, v)
    del position 
    x_range = torch.linspace(ARGS.x_min, ARGS.x_max, ARGS.x_samples, device=device)
    y_range = torch.linspace(ARGS.y_min, ARGS.y_max, ARGS.y_samples, device=device)

    for i in range(ARGS.x_begin, ARGS.x_end):
        for j in range(ARGS.y_begin, ARGS.y_end):
            print_fn('Sweep {}, {}'.format(i, j))
            cu = x_range[i]
            cv = y_range[j]
            with torch.no_grad():
                shift = (cu-cu0) * u + (cv-cv0) * v

            train_loss, train_top1, test_loss, test_top1 = shift_and_eval(
                model, loss, train_loader, test_loader,
                device, shift, verbose=True
            )

            save_dict = {
                "grid_coordinates": (i,j),
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_top1": train_top1,
                "test_top1": test_top1,
            }
            filename = f"{save_path}/{i}_{j}.h5"
            if ARGS.tpu:
                if xm.get_ordinal() == 0 and filename[0:5] == "gs://":
                    from utils.gcloud import post_file_to_bucket

                    dd.io.save(filename, save_dict)
                    post_file_to_bucket(filename)
            
            model = load.model(ARGS.model, ARGS.model_class)(
                input_shape=input_shape, num_classes=num_classes, pretrained=ARGS.pretrained,
                model_dir=ARGS.model_dir,
            )
            if len(ARGS.gpu.split(",")) > 1:
                model = nn.DataParallel(model)
            model = model.to(device)

if __name__ == "__main__":
    parser = flags.extract()
    parser = extend_parser(parser)
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
