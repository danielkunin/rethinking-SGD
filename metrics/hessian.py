from tqdm import tqdm
import numpy as np
import deepdish as dd


def load_weight_and_grad(step, feats_dir):
    load_path = f"{feats_dir}/step{step}.h5"
    weight = dd.io.load(load_path, f"/position")
    grad = dd.io.load(load_path, f"/velocity")
    return weight, grad


def load_eigenvalues(metrics_dir):
    spectral_cache = f"{metrics_dir}/metrics/spectral.h5"
    eigenvectors = dd.io.load(spectral_cache, f"/eigenvector")
    eigenvalues = dd.io.load(spectral_cache, f"/eigenvalues")
    eigen_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigen_idx]
    eigenvectors = eigenvectors[:,eigen_idx]
    return eigenvalues, eigenvectors


def hessian_eigenprojection(model, feats_dir, steps, **kwargs):
    eigenvalues, eigenvectors = load_eigenvalues(feats_dir.split("/feats")[0])

    projected_weights = np.zeros((len(steps), eigenvalues.shape[0]))
    projected_grads = np.zeros((len(steps), eigenvalues.shape[0]))

    for i in tqdm(range(0, len(steps))):
        step = steps[i]
        weight, grad = load_weight_and_grad(step, feats_dir)
        projected_weights[i] = eigenvectors.T @ weight
        projected_grads[i] = eigenvectors.T @ grad


    projected_weights_and_grads = {
        "steps": np.array(steps),
        "projected_weights": projected_weights,
        "projected_grads": projected_grads,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
    }

    return projected_weights_and_grads

def compute_fft(time_series):
    ft = np.fft.rfft(time_series, axis=0)
    freq = np.fft.rfftfreq(len(time_steps))
    fft = {
        "amplitude": np.array([a.real for a in ft]),
        "frequency": freq,
    }
    return fft

def fft(model, feats_dir, steps, **kwargs):
    eigenvalues, eigenvectors = load_eigenvalues(feats_dir.split("/feats")[0])

    projected_weights = np.zeros((len(steps), eigenvalues.shape[0]))
    projected_grads = np.zeros((len(steps), eigenvalues.shape[0]))

    for i in tqdm(range(0, len(steps))):
        step = steps[i]
        weight, grad = load_weight_and_grad(step, feats_dir)
        projected_weights[i] = eigenvectors.T @ weight
        projected_grads[i] = eigenvectors.T @ grad

    fft_vals = {
        "steps": np.array(steps),
        "weight_fft": compute_fft(projected_weights),
        "grad_fft": compute_fft(projected_grads),
        "weight_grad_fft": compute_fft(np.concatenate([projected_weights, projected_grads], axis=1)),
    }

    return fft_vals
