import torch
import deepdish as dd
import numpy as np
from tqdm import tqdm


def checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    curr_step,
    save_path,
    verbose,
    metric_dict={},
    tpu=False,
    lean=False,
):
    save_lib = torch
    print_fn = print
    if tpu:
        import torch_xla.core.xla_model as xm

        save_lib = xm
        print_fn = xm.master_print

    if verbose:
        print_fn(f"Saving model checkpoint for step {curr_step}")
    save_dict = {
        "epoch": epoch,
        "step": curr_step
    }
    save_dict.update(metric_dict)
    if not lean:
        save_dict.update({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        })
    filename = f"{save_path}/ckpt/step{curr_step}.tar"
    save_lib.save(
        save_dict, filename,
    )
    if tpu:
        if xm.get_ordinal() == 0 and filename[0:5] == "gs://":
            from utils.gcloud import post_file_to_bucket

            post_file_to_bucket(filename, verbose)


# TODO: we maybe don't want to have the scheduler inside the train function
def train(
    model,
    loss,
    optimizer,
    scheduler,
    dataloader,
    device,
    epoch,
    verbose,
    save,
    save_freq,
    save_begin_epoch,
    save_path,
    log_interval=10,
    lean_ckpt=False,
    test_loader=None,
    **kwargs,
):
    batch_size = kwargs.get("batch_size")  # per core batch size
    num_batches = kwargs.get("num_batches")  #  len(dataloader)
    dataset_size = kwargs.get("dataset_size")  # len(dataloader.dataset)

    print_fn = print
    if device.type == "xla":
        import torch_xla.core.xla_model as xm

        xrt_world_size = kwargs.get("xrt_world_size")
        xm_ordinal = kwargs.get("xm_ordinal")
        tracker = xm.RateTracker()
        if verbose <= 1:
            print_fn = xm.master_print

    model.train()
    total_loss = 0
    total_samples = 0
    correct1 = 0
    correct5 = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        curr_step = epoch * num_batches + batch_idx
        ###### Batch loading
        if device.type != "xla":
            data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total_loss += train_loss.item() * data.size(0)
        total_samples += data.size(0)
        train_loss.backward()
        if device.type == "xla":
            xm.optimizer_step(optimizer)
            tracker.add(batch_size)
        else:
            optimizer.step()
        curr_step += 1

        # Train accuracy
        _, pred = output.topk(5, dim=1)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        correct1 += correct[:, :1].sum().item()
        correct5 += correct[:, :5].sum().item()

        ###### Logging
        if verbose and (batch_idx % log_interval == 0):
            examples_seen = batch_idx * batch_size
            per_worker_header = ""
            if device.type == "xla" and verbose >= 2:
                per_worker_header = (
                    f"[xla:{xm_ordinal}, "
                    f"rate: {tracker.rate():.2f}, "
                    f"global_rate: {tracker.global_rate():.2f}]\t"
                )
                examples_seen *= xrt_world_size
                examples_seen += xm_ordinal * batch_size
            print_fn(
                f"{per_worker_header}"
                f"Train Epoch: {epoch} "
                f"[{examples_seen}/{dataset_size} "
                f"({100.0*batch_idx/num_batches:.0f}%)]"
                f"\tLoss: {train_loss.item():.6f}"
                f"\tStep: {curr_step}"
            )

        # TODO: this is just to be able to save at any step (even mid-epoch)
        #       it might make more sense to checkpoint only on epoch: makes
        #       for a cleaner codebase and can include test metrics
        # TODO: additionally, could integrate tfutils.DBInterface here
        # TODO: customize the metric dict based on flags
        ######## Checkpointing
        if save and save_path is not None and save_freq is not None:
            # Do this for consecutive steps
            if curr_step % save_freq <= 0 and (epoch + batch_idx/num_batches) >= save_begin_epoch:
                pos, vel = optimizer.track()
                metric_dict = {
                    "pos_norm": torch.norm(pos),
                    "vel_norm": torch.norm(vel),
                    "pos_dot": torch.dot(pos, kwargs["theta_0"]),
                    "dist_from_start": torch.norm(pos - kwargs["theta_0"]),
                }
                if "eigenvectors" in kwargs.keys():
                    metric_dict["projected_pos"] = torch.matmul(kwargs["eigenvectors"], pos),
                    metric_dict["projected_vel"] = torch.matmul(kwargs["eigenvectors"], vel),
                if kwargs["eval_mid_epoch"]:
                    test_loss, test_accuracy1, test_accuracy5 = eval(
                        model, loss, test_loader, device, verbose, epoch
                    )
                    model.train()
                    eval_metrics = {
                        "train_loss": train_loss.item(),
                        "train_batch_accuracy1": correct[:, :1].sum().item(),
                        "train_batch_accuracy5": correct[:, :5].sum().item(),
                        "test_loss": test_loss,
                        "test_accuracy1": test_accuracy1,
                        "test_accuracy5": test_accuracy5,
                    }
                    metric_dict.update(eval_metrics)

                checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    curr_step,
                    save_path,
                    verbose,
                    metric_dict=metric_dict,
                    tpu=(device.type == "xla"),
                    lean=lean_ckpt,
                )

    if device.type == "xla":
        total_loss = xm.mesh_reduce("total_train_loss", total_loss, np.sum)
        total_samples = xm.mesh_reduce("total_train_samples", total_samples, np.sum)
        correct1 = xm.mesh_reduce("total_train_correct1", correct1, np.sum)
        correct5 = xm.mesh_reduce("total_train_correct5", correct5, np.sum)
    average_loss = 1.0 * total_loss / total_samples
    accuracy1 = 100.0 * correct1 / total_samples
    accuracy5 = 100.0 * correct5 / total_samples
    return average_loss, accuracy1, accuracy5


def eval(model, loss, dataloader, device, verbose, epoch, **kwargs):
    print_fn = print
    if device.type == "xla":
        import torch_xla.core.xla_model as xm

        print_fn = xm.master_print

    model.eval()
    total_loss = 0
    correct1 = 0
    correct5 = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:, :1].sum().item()
            correct5 += correct[:, :5].sum().item()
            total_samples += data.size()[0]

    if device.type == "xla":
        total_loss = xm.mesh_reduce("total_test_loss", total_loss, np.sum)
        total_samples = xm.mesh_reduce("total_test_samples", total_samples, np.sum)
        correct1 = xm.mesh_reduce("total_test_correct1", correct1, np.sum)
        correct5 = xm.mesh_reduce("total_test_correct5", correct5, np.sum)

    average_loss = 1.0 * total_loss / total_samples
    accuracy1 = 100.0 * correct1 / total_samples
    accuracy5 = 100.0 * correct5 / total_samples
    print_fn(
        f"Epoch {epoch} evaluation: Average Test Loss: {average_loss:.4f}, "
        f"Top 1 Test Accuracy: {correct1}/{total_samples} ({accuracy1:.2f}%)"
    )
    return average_loss, accuracy1, accuracy5


def train_eval_loop(
    model,
    loss,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    device,
    epochs,
    verbose,
    save,
    save_freq=None,
    save_begin_epoch=0,
    save_path=None,
    epoch_offset=0,
    lean_ckpt=False,
    **kwargs,
):
    print_fn = print
    if device.type == "xla":
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.core.xla_model as xm

        print_fn = xm.master_print
        train_loader = pl.MpDeviceLoader(train_loader, device)
        test_loader = pl.MpDeviceLoader(test_loader, device)

    # Get the weights at initialization
    trainabe_weights = []
    for name,param in model.named_parameters():
        if param.requires_grad:
            trainabe_weights.append(param.detach().clone())
    theta_0 = torch.cat([p.reshape(-1) for p in trainabe_weights])
    kwargs["theta_0"] = theta_0

    # Also get the eigenvectors if a path is specified
    if kwargs["spectral_path"]:
        print_fn("Including evecs in kwargs")
        evecs = dd.io.load(kwargs["spectral_path"], "/eigenvector")
        kwargs["eigenvectors"] = torch.tensor(evecs.T, device=device)

    # Initial eval
    test_loss, test_accuracy1, test_accuracy5 = eval(model, loss, test_loader, device, verbose, 0)
    metric_dict = {
        "train_loss": 0,
        "test_loss": test_loss,
        "test_accuracy1": test_accuracy1,
        "test_accuracy5": test_accuracy5,
    }
    if save:
        checkpoint(
            model,
            optimizer,
            scheduler,
            0,
            0,
            save_path,
            verbose,
            metric_dict,
            tpu=(device.type == "xla"),
        )
    for epoch in tqdm(range(epoch_offset, epoch_offset + epochs)):
        train_loss, train_accuracy1, train_accuracy5 = train(
            model,
            loss,
            optimizer,
            scheduler,
            train_loader,
            device,
            epoch,
            verbose,
            save,
            save_freq=save_freq,
            save_begin_epoch=save_begin_epoch,
            save_path=save_path,
            lean_ckpt=lean_ckpt,
            test_loader=test_loader,
            **kwargs,
        )
        print_fn(
            f"Epoch {epoch}: Average Train Loss: {train_loss:.4f}, "
            f"Top 1 Train Accuracy: {train_accuracy1:.2f}%"
        )
        test_loss, test_accuracy1, test_accuracy5 = eval(
            model, loss, test_loader, device, verbose, epoch + 1
        )
        metric_dict = {
            "train_loss": train_loss,
            "train_accuracy1": train_accuracy1,
            "train_accuracy5": train_accuracy5,
            "test_loss": test_loss,
            "test_accuracy1": test_accuracy1,
            "test_accuracy5": test_accuracy5,
        }
        curr_step = (epoch + 1) * kwargs.get("num_batches")
        if save:
            checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                curr_step,
                save_path,
                verbose,
                metric_dict,
                tpu=(device.type == "xla"),
            )
        scheduler.step()
    if epochs > 0:
        print_fn(
            f"Final performance: "
            f"\tTrain Loss: {train_loss:.4f}"
            f"\tTest Loss: {test_loss:.4f}"
            f"\tTest Accuracy: {test_accuracy1:.2f}%"
        )
