import os
import torch
import numpy as np
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from models import mlp
from models import tinyimagenet_vgg
from models import tinyimagenet_resnet
from models import tinyimagenet_alexnet
from models import tinyimagenet_densenet
from models import tinyimagenet_googlenet
from models import imagenet_vgg
from models import imagenet_resnet
from models import imagenet_alexnet
from models import imagenet_densenet
from models import imagenet_googlenet
from optimizers import custom_sgd, neg_momentum_sgd
from utils import custom_datasets


def configure_tpu(tpu_name):
    from utils.gcloud import lookup_tpu_ip_by_name, configure_env_for_tpu

    print("Configuring ENV variables for TPU training")
    tpu_ip = lookup_tpu_ip_by_name(tpu_name)
    configure_env_for_tpu(tpu_ip)


def device(gpu, tpu=None):
    if tpu:
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        print(f"CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}")
        use_cuda = torch.cuda.is_available()
        return torch.device("cuda" if use_cuda else "cpu")


def MSELoss(output, target, reduction='mean'):
    num_classes = output.size(1)
    labels = F.one_hot(target, num_classes=num_classes)
    if reduction is 'mean':
        return torch.mean(torch.sum((output - labels)**2, dim=1))/2
    elif reduction is 'sum':
        return torch.sum((output - labels)**2)/2
    elif reduction is None:
        return ((output - labels)**2)/2
    else:
        raise ValueError(reduction + " is not valid")


def loss(name):
    losses = {
        "mse": MSELoss,
        "ce": torch.nn.CrossEntropyLoss()
    }
    return losses[name]


def dimension(dataset):
    if dataset == "mnist":
        input_shape, num_classes = (1, 28, 28), 10
    if dataset == "cifar10":
        input_shape, num_classes = (3, 32, 32), 10
    if dataset == "cifar100":
        input_shape, num_classes = (3, 32, 32), 100
    if dataset == "tiny-imagenet":
        input_shape, num_classes = (3, 64, 64), 200
    if dataset == "imagenet":
        input_shape, num_classes = (3, 224, 224), 1000
    return input_shape, num_classes


def get_transform(size, padding, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(transforms.RandomCrop(size=size, padding=padding))
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean, std))
    return transforms.Compose(transform)


def dataloader(
    dataset, batch_size, train, workers, length=None, datadir="Data", tpu=False, 
    shuffle=True, data_augment=True,
):
    # Dataset
    if dataset == "mnist":
        mean, std = (0.1307,), (0.3081,)
        transform = get_transform(
            size=28, padding=0, mean=mean, std=std, preprocess=False
        )
        dataset = datasets.MNIST(
            datadir, train=train, download=True, transform=transform
        )
    if dataset == "cifar10":
        mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
        transform = get_transform(
            size=32, padding=4, mean=mean, std=std, preprocess=train
        )
        dataset = datasets.CIFAR10(
            datadir, train=train, download=True, transform=transform
        )
    if dataset == "cifar100":
        mean, std = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
        transform = get_transform(
            size=32, padding=4, mean=mean, std=std, preprocess=train
        )
        dataset = datasets.CIFAR100(
            datadir, train=train, download=True, transform=transform
        )
    if dataset == "tiny-imagenet":
        mean, std = (0.480, 0.448, 0.397), (0.276, 0.269, 0.282)
        transform = get_transform(
            size=64, padding=4, mean=mean, std=std, preprocess=train
        )
        dataset = custom_datasets.TINYIMAGENET(
            datadir, train=train, download=True, transform=transform
        )
    if dataset == "imagenet":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if train and data_augment:
            # Torch pretrained models do not use grayscale, jitter, nor change the scale of crops
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224), #, scale=(0.2, 1.0)),
                    #transforms.RandomGrayscale(p=0.2),
                    #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        folder = f"{datadir}/imagenet_raw/{'train' if train else 'val'}"
        dataset = datasets.ImageFolder(folder, transform=transform)

    # Dataloader
    shuffle = (train is True) and shuffle
    if length is not None:
        indices = torch.randperm(len(dataset))[:length]
        dataset = torch.utils.data.Subset(dataset, indices)

    sampler = None
    kwargs = {}
    if torch.cuda.is_available():
        kwargs = {"num_workers": workers, "pin_memory": True}
    elif tpu:
        import torch_xla.core.xla_model as xm

        # TODO: might want to drop last to keep batches the same size and
        # speed up computation
        kwargs = {"num_workers": workers}  # , "drop_last": True}
        if xm.xrt_world_size() > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=shuffle,
            )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False if sampler else shuffle,
        sampler=sampler,
        **kwargs,
    )

    return dataloader


def model(model_architecture, model_class):
    default_models = {
        "logistic": mlp.logistic,
        "fc": mlp.fc,
        "fc-bn": mlp.fc_bn,
        "conv": mlp.conv,
    }
    tinyimagenet_models = {
        "vgg11": tinyimagenet_vgg.vgg11,
        "vgg11-bn": tinyimagenet_vgg.vgg11_bn,
        "vgg13": tinyimagenet_vgg.vgg13,
        "vgg13-bn": tinyimagenet_vgg.vgg13_bn,
        "vgg16": tinyimagenet_vgg.vgg16,
        "vgg16-bn": tinyimagenet_vgg.vgg16_bn,
        "vgg19": tinyimagenet_vgg.vgg19,
        "vgg19-bn": tinyimagenet_vgg.vgg19_bn,
        "resnet18": tinyimagenet_resnet.resnet18,
        "resnet34": tinyimagenet_resnet.resnet34,
        "resnet50": tinyimagenet_resnet.resnet50,
        "resnet101": tinyimagenet_resnet.resnet101,
        "resnet152": tinyimagenet_resnet.resnet152,
        "wide-resnet18": tinyimagenet_resnet.wide_resnet18,
        "wide-resnet34": tinyimagenet_resnet.wide_resnet34,
        "wide-resnet50": tinyimagenet_resnet.wide_resnet50,
        "wide-resnet101": tinyimagenet_resnet.wide_resnet101,
        "wide-resnet152": tinyimagenet_resnet.wide_resnet152,
        "resnet18-nobn": tinyimagenet_resnet.resnet18_nobn,
        "resnet34-nobn": tinyimagenet_resnet.resnet34_nobn,
        "resnet50-nobn": tinyimagenet_resnet.resnet50_nobn,
        "resnet101-nobn": tinyimagenet_resnet.resnet101_nobn,
        "resnet152-nobn": tinyimagenet_resnet.resnet152_nobn,
        "wide-resnet18-nobn": tinyimagenet_resnet.wide_resnet18_nobn,
        "wide-resnet34-nobn": tinyimagenet_resnet.wide_resnet34_nobn,
        "wide-resnet50-nobn": tinyimagenet_resnet.wide_resnet50_nobn,
        "wide-resnet101-nobn": tinyimagenet_resnet.wide_resnet101_nobn,
        "wide-resnet152-nobn": tinyimagenet_resnet.wide_resnet152_nobn,
        "alexnet": tinyimagenet_alexnet.alexnet,
        "densenet121": tinyimagenet_densenet.densenet121,
        "densenet161": tinyimagenet_densenet.densenet161,
        "densenet169": tinyimagenet_densenet.densenet169,
        "densenet201": tinyimagenet_densenet.densenet201,
        "googlenet": tinyimagenet_googlenet.googlenet,
    }
    imagenet_models = {
        "vgg11": imagenet_vgg.vgg11,
        "vgg11-bn": imagenet_vgg.vgg11_bn,
        "vgg13": imagenet_vgg.vgg13,
        "vgg13-bn": imagenet_vgg.vgg13_bn,
        "vgg16": imagenet_vgg.vgg16,
        "vgg16-bn": imagenet_vgg.vgg16_bn,
        "vgg19": imagenet_vgg.vgg19,
        "vgg19-bn": imagenet_vgg.vgg19_bn,
        "resnet18": imagenet_resnet.resnet18,
        "resnet34": imagenet_resnet.resnet34,
        "resnet50": imagenet_resnet.resnet50,
        "resnet101": imagenet_resnet.resnet101,
        "resnet152": imagenet_resnet.resnet152,
        "wide-resnet50": imagenet_resnet.wide_resnet50_2,
        "wide-resnet101": imagenet_resnet.wide_resnet101_2,
        "alexnet": imagenet_alexnet.alexnet,
        "densenet121": imagenet_densenet.densenet121,
        "densenet161": imagenet_densenet.densenet161,
        "densenet169": imagenet_densenet.densenet169,
        "densenet201": imagenet_densenet.densenet201,
        "googlenet": imagenet_googlenet.googlenet,

    }
    models = {
        "default": default_models,
        "tinyimagenet": tinyimagenet_models,
        "imagenet": imagenet_models,
    }
    return models[model_class][model_architecture]


def optimizer(optimizer, momentum=0.0, dampening=0.0, nesterov=False):
    optimizers = {
        "custom_sgd": (
            custom_sgd.SGD,
            {
                "momentum": momentum,
                "dampening": dampening,
                "nesterov": nesterov,
            },
        ),
        "sgd": (optim.SGD, {}),
        "neg_momentum": (
            neg_momentum_sgd.SGD,
            {
                "momentum": momentum,
                "dampening": dampening,
                "nesterov": nesterov,
            },
        ),
        "momentum": (
            optim.SGD,
            {"momentum": momentum, "dampening": dampening, "nesterov": nesterov},
        ),
        "adam": (optim.Adam, {}),
        "rms": (optim.RMSprop, {}),
    }
    return optimizers[optimizer]
