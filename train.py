import argparse
import os
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# Makes folders for storing results
def mkdir(dir, overwrite=False):
    if not os.path.exists(dir):
        os.makedirs(dir)
    elif overwrite:
        shutil.rmtree(dir)
        os.makedirs(dir)

# Computes Hessian vector product
def Hvp(loss, v, model, device, data_loader):
    Hv = torch.zeros_like(v)
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        L = loss(output, target, reduction='sum') / len(data_loader.dataset)
        grad = torch.autograd.grad(L, model.parameters(), create_graph=True)
        grad_vec = torch.cat([g.reshape(-1) for g in grad if g is not None])
        prod = torch.dot(grad_vec, v)
        grad = torch.autograd.grad(prod, model.parameters(), retain_graph=True)
        Hv += torch.cat([g.reshape(-1) for g in grad if g is not None])
    return Hv

# Computes top eigensubspace of Hessian via power series
def subspace(loss, model, device, data_loader, dim, iters):
    model.train()
    m = sum(p.numel() for p in model.parameters())
    Q = torch.randn((m, dim)).to(device)
    for i in tqdm(range(iters)):
        HV = torch.zeros((m, dim))
        for j in tqdm(range(dim), leave=False):
            HV[:,j] = Hvp(loss, Q[:,j], model, device, data_loader)
        Q, R = torch.qr(HV)
    return Q.data.numpy(), torch.diag(R).data.numpy()

# Computes complete Hessian matrix
def hessian(loss, model, device, data_loader):
    model.train()
    m = sum(p.numel() for p in model.parameters())
    H = torch.zeros(m, m)
    for i in tqdm(range(m)):
        v = torch.zeros(m)
        v[i] = 1.0
        H[i] = Hvp(loss, v, model, device, data_loader)
    return H.data.numpy()

# Extends SGD optimizer with tracking and upate functionality
class SGD(optim.SGD):
    @torch.no_grad()
    def track(self):
        position = []
        velocity = []
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if "momentum_buffer" in param_state:
                    buf = param_state["momentum_buffer"]
                else:
                    buf = torch.zeros_like(p)
                position.append(p.data.numpy().flatten())
                velocity.append(buf.data.numpy().flatten())
        return np.concatenate(position), np.concatenate(velocity)

    def update(self, name, value):
        for group in self.param_groups:
            group[name] = value

    def reset(self):
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["momentum_buffer"].mul_(0.0)

# Builds fully connected models of certain depth, width, nonlinearity
def fc(args, input_shape, num_classes, nonlinearity=nn.ReLU()):
    input_dim = np.prod(input_shape)
    modules = [nn.Flatten()]
    if args.num_layers == 0:
        modules.append(nn.Linear(input_dim, num_classes))
    else:
        modules.append(nn.Linear(input_dim, args.num_units))
        if args.batchnorm:
                modules.append(nn.BatchNorm1d(args.num_units))
        modules.append(nonlinearity)
        for i in range(args.num_layers - 1):
            modules.append(nn.Linear(args.num_units, args.num_units))
            if args.batchnorm:
                modules.append(nn.BatchNorm1d(args.num_units))
            modules.append(nonlinearity)
        modules.append(nn.Linear(args.num_units, num_classes))
    model = nn.Sequential(*modules)
    if args.pretrained:
        pretrained_path = "{}/{}/model.pt".format(args.save_dir, args.expid)
        model.load_state_dict(torch.load(pretrained_path))
    return model

# Builds convolution models of certain depth, filters, nonlinearity
def conv(args, input_shape, num_classes, nonlinearity=nn.ReLU()):
    channels, width, height = input_shape
    modules = []
    if args.num_layers == 0:
        modules.append(nn.Flatten())
        modules.append(nn.Linear(input_dim, num_classes))
    else:
        modules.append(nn.Conv2d(channels, args.num_units, kernel_size=3, padding=3 // 2))
        if args.batchnorm:
            modules.append(nn.BatchNorm2d(args.num_units))
        modules.append(nonlinearity)
        for i in range(args.num_layers - 1):
            modules.append(nn.Conv2d(args.num_units, args.num_units, kernel_size=3, padding=3 // 2))
            if args.batchnorm:
                modules.append(nn.BatchNorm2d(args.num_units))
            modules.append(nonlinearity)
        modules.append(nn.Flatten())
        modules.append(nn.Linear(args.num_units * width * height, num_classes))
    model = nn.Sequential(*modules)
    if args.pretrained:
        pretrained_path = "{}/{}/model.pt".format(args.save_dir, args.expid)
        model.load_state_dict(torch.load(pretrained_path))
    return model

# Train loop    
def train(args, loss, model, device, train_loader, optimizer, epoch, step):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.track & (step % args.track_interval == 0):
            position, velocity = optimizer.track()
            np.save("{}/{}/position/{}.npy".format(args.save_dir, args.expid, step), position)
            np.save("{}/{}/velocity/{}.npy".format(args.save_dir, args.expid, step), velocity)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        L = loss(output, target)
        L.backward()
        optimizer.step()
        step += 1
        if args.verbose & (batch_idx % args.log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), L.item()))
    return step

# Test loop   
def test(args, loss, model, device, test_loader):
    model.eval()
    test_L = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_L += loss(output, target, reduction='sum').item()
            estimate = output.argmax(dim=1, keepdim=True)
            correct += estimate.eq(target.view_as(estimate)).sum().item()
    test_L /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    if args.verbose:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_L, correct, len(test_loader.dataset),accuracy))
    return accuracy

# Returns Torch dataloader
def dataloader(dataset, bs, kwargs, sampler=None):
    shuffle = sampler is None
    return torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=shuffle, **kwargs)

# Anneal hyperparameter function
def anneal(lr, mom, bs, alpha):
    # # Strategy 0
    # _bs = int(60000)
    # _mom = 1.0
    # _lr = lr
    # # Strategy 1
    # _bs = int(bs / alpha)
    # _mom = np.sqrt(1 - alpha * (1 - mom**2))
    # _lr = lr * (1 + mom) / (1 + _mom)
    # Strategy 2
    # _lr = lr * alpha
    # _mom = ((1 + mom) - alpha * (1 - mom)) / ((1 + mom) + alpha * (1 - mom))
    # _bs = bs
    # Strategy 3
    _bs = bs
    _lr = lr * alpha
    _mom = 1 - 2*_lr
    return _lr, _mom, _bs

# Custom MSE loss for classification
def MSELoss(output, target, reduction='mean'):
    num_classes = output.size(1)
    labels = F.one_hot(target, num_classes=num_classes)
    if reduction is 'mean':
        return torch.mean(torch.sum((output - labels)**2, axis=1)) / 2
    elif reduction is 'sum':
        return torch.sum((output - labels)**2) / 2
    elif reduction is None:
        return (output - labels)**2 / 2
    else:
        raise ValueError(reduction + " is not valid")

# Custom CE loss for classification
def CELoss(output, target, reduction='mean'):
    return F.cross_entropy(output, target, reduction=reduction)

# Disctionary of losses
losses = {
    "mse": MSELoss,
    "ce": CELoss
}

def main():
    parser = argparse.ArgumentParser(description='Neural Mechanics II')
    # Hyperparameters
    parser.add_argument('--bs', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-bs', type=int, default=1024,
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--mom', type=float, default=0.9,
                        help='Momentum coeficient (default: 0.9)')
    parser.add_argument('--reg', type=float, default=0.0,
                        help='L2 regularization constant (default: 0.0)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train at convergence (default: 10)')
    parser.add_argument('--drop-rate', type=float, default=0.1,
                        help='multiplicative factor for drop (default: 0.1)')
    parser.add_argument('--drops', type=int, nargs='*', default=[],
                        help='List of epochs to apply hyperparameter drops (default: [])')
    # Loss & Model
    parser.add_argument('--loss', type=str, default='ce',
                        help='mean squared error (mse) or cross entropy (ce)')
    parser.add_argument('--model', type=str, default='fc', choices=["fc", "conv"],
                        help='Model type (fc or conv)')
    parser.add_argument('--num-layers', type=int, default=0,
                        help='how many hidden layers in model')
    parser.add_argument('--num-units', type=int, default=100,
                        help='how many hidden neurons per layer in model')
    parser.add_argument('--batchnorm', type=bool, default=False,
                        help='Apply batchnormalization to hidden layers')
    parser.add_argument('--pretrained', type=bool, default=False,
                        help='Load pretrained model')
    # Metrics
    parser.add_argument('--track', type=bool, default=False,
                        help='track position and velocity during training')
    parser.add_argument('--track-interval', type=int, default=1,
                        help='how many batches to wait before tracking position and velocity')
    parser.add_argument('--eigenvector', type=bool, default=False,
                        help='Compute top eigenvector')
    parser.add_argument('--eigen-dims', type=int, default=1,
                        help='dimension of eigenspace to compute')
    parser.add_argument('--power-iters', type=int, default=5,
                        help='Number of iterations of power series to use')
    parser.add_argument('--hessian', type=bool, default=False,
                        help='Compute and save Hessian matrix')
    # Experiment
    parser.add_argument('--verbose', type=bool, default=False,
                        help='print statistics during training and testing')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--save-dir', type=str, default='results',
                        help='Directory to save results and model')
    parser.add_argument('--expid', type=str, required=True,
                        help='Directory to save experiment within save directory')
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    if args.track:
        mkdir("{}/{}/position".format(args.save_dir, args.expid), overwrite=True)
        mkdir("{}/{}/velocity".format(args.save_dir, args.expid), overwrite=True)
    
    # MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('data', train=False, transform=transform)
    train_loader = dataloader(train_data, args.bs, kwargs)
    test_loader = dataloader(test_data, args.test_bs, kwargs)

    # Get task, model, and optimizer
    loss = losses[args.loss]
    input_shape, num_classes = (1, 28, 28), 10
    if args.model == 'fc':
        model = fc(args, input_shape, num_classes).to(device)
    if args.model == 'conv':
        model = conv(args, input_shape, num_classes).to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.reg)

    # Train model
    lr, mom, bs = args.lr, args.mom, args.bs
    accuracy = []
    step = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        accuracy.append(test(args, loss, model, device, test_loader))
        if epoch in args.drops:
            lr, mom, bs = anneal(lr, mom, bs, args.drop_rate)
            optimizer.update('lr', lr)
            optimizer.update('momentum', mom)
            train_loader = dataloader(train_data, bs, kwargs) # torch.RandomSampler(data,True)?
            if args.verbose:
                print("Learning rate: {}, Momentum: {}, Batch size: {}".format(lr, mom, bs))
        step = train(args, loss, model, device, train_loader, optimizer, epoch, step)
    accuracy.append(test(args, loss, model, device, test_loader))

    # Metrics and Save
    if args.eigenvector:
        print("Computing Eigenvector")
        data_loader = train_loader#dataloader(train_data, 60000, kwargs)
        V, Lamb = subspace(loss, model, device, data_loader, args.eigen_dims, args.power_iters)
        np.save("{}/{}/eigenvector.npy".format(args.save_dir, args.expid), V)
        np.save("{}/{}/eigenvalues.npy".format(args.save_dir, args.expid), Lamb)
    if args.hessian:
        print("Computing Hessian")
        data_loader = train_loader#dataloader(train_data, 60000, kwargs)
        H = hessian(loss, model, device, data_loader)
        np.save("{}/{}/hessian.npy".format(args.save_dir, args.expid), H)
    if args.save_model:
        np.save("{}/{}/accuracy.npy".format(args.save_dir, args.expid), np.array(accuracy))
        torch.save(model.state_dict(),"{}/{}/model.pt".format(args.save_dir, args.expid))
        
if __name__ == '__main__':
    main()

# QUESTIONS/COMMENTS:
# (1) I think its more efficient to compute the higher-order gradients per batch 
# rather than over the full gradient that way less is stored in memory, but not sure. Note,
# the loop over data can be parallelized as the model doesnt change between batches.
# (2) How can we rewite the eigenvector function to gives us top-k eigenvector/value pairs?
# (3) Saving the momentum buffer or negative momentum buffer will determine whether 
#  osccilations are clockwise or counter-clockwise. Which one makes more sense?
# (4) What is the cleanest way to generalizer all annealing strategies?
# (5) Best strategy for changing batch size?
# (6) How does batchnormalization change things? Should these parameters be counted as well?
