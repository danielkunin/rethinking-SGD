import torch
import numpy as np
import deepdish as dd
from tqdm import tqdm
from scipy.sparse.linalg import LinearOperator, eigsh

# Computes Hessian vector product
def Hvp(loss, v, model, device, data_loader):
    Hv = torch.zeros_like(v, device=device)
    for batch_idx, (data, target) in tqdm(enumerate(data_loader), desc='Dataset', leave=True):
        data, target = data.to(device), target.to(device)
        output = model(data)
        L = loss(output, target) * data.size(0) / len(data_loader.dataset)
        grad = torch.autograd.grad(L, model.parameters(), create_graph=True)
        grad_vec = torch.cat([g.reshape(-1) for g in grad if g is not None])
        prod = torch.dot(grad_vec, v)
        grad = torch.autograd.grad(prod, model.parameters())
        Hv += torch.cat([g.reshape(-1) for g in grad if g is not None])
    if device.type == "xla":
        import torch_xla.core.xla_model as xm
        Hv = xm.mesh_reduce("Hv", Hv, np.sum)
    return Hv

# Computes top eigensubspace of Hessian via power iteration
def subspace(loss, model, device, data_loader, dim, iters, save_path):
    model.train()
    m = sum(p.numel() for p in model.parameters())
    Q = torch.randn((m, dim), device=device)
    for i in tqdm(range(iters), desc='Power iteration', leave=True):
        HV = torch.zeros((m, dim), device=device)
        for j in tqdm(range(dim), desc='Eigenvector', leave=True):
            HV[:,j] = Hvp(loss, Q[:,j], model, device, data_loader)
        Q, R = torch.qr(HV)
        Q = Q.to(device)
        R = R.to(device)
        V = Q.data.cpu().numpy()
        lamb =  torch.diag(R).data.cpu().numpy()
        dd.io.save(
            f"{save_path}/spectral_it{i}.h5",
            {"eigenvector": V, "eigenvalues": lamb}
        )
    return V, lamb
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

# The following is based on: https://github.com/locuslab/edge-of-stability/blob/bec8069239d23d55bdb7f87456a844b5d8f24445/src/utilities.py
# It has been modified to compute top and bottom evals
def lanczos(matrix_vector, dim: int, neigs: int):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).cuda()
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    # Possible values for which: LM | SM | LA | SA | BE
    # Also note fro docs: . If small eigenvalues are desired,
    # consider using shift-invert mode for better performance.
    evals, evecs = eigsh(operator, neigs, which="BE")
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()


def get_hessian_eigenvalues(loss, model, device, data_loader, neigs=6):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: Hvp(loss, delta, model,
                                  device, data_loader).detach().cpu()
    nparams = sum(p.numel() for p in model.parameters())
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return evals, evecs


def gradient(loss, model, device, data_loader):
    model.train()
    m = sum(p.numel() for p in model.parameters())
    gradient = torch.zeros(m, device=device)
    for batch_idx, (data, target) in tqdm(enumerate(data_loader), desc='Dataset', leave=True):
        data, target = data.to(device), target.to(device)
        output = model(data)
        L = loss(output, target) * data.size(0) / len(data_loader.dataset)
        grad = torch.autograd.grad(L, model.parameters())
        grad_vec = torch.cat([g.reshape(-1) for g in grad if g is not None])
        gradient += grad_vec
    if device.type == "xla":
        import torch_xla.core.xla_model as xm
        gradient = xm.mesh_reduce("gradient", gradient, np.sum)
    return gradient

