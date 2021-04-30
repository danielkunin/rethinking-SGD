import os
import numpy as np
import pprint
import h5py


def makedir_quiet(d):
    """
    Convenience util to create a directory if it doesn't exist
    """
    if not os.path.isdir(d):
        os.makedirs(d)


def make_iterable(x):
    """
    If x is not already array_like, turn it into a list or np.array
    """
    if not isinstance(x, (list, tuple, np.ndarray)):
        return [x]
    return x


def get_features(
    feats_path, group, keys, out_keys=None, verbose=False,
):
    """
    Returns features from HDF5 DataSet

    Inputs
        validation_path (str): where to find the HDF5 dataset
        group_name (str): the group name used for the particular validation
        keys (str or list of strs): which keys to extract from the group.
        out_keys (list of strs): keys for the output dict
    """
    assert os.path.isfile(feats_path), f"{feats_path} is not a file"

    keys = make_iterable(keys)

    if out_keys is None:
        out_keys = keys
    out_keys = make_iterable(out_keys)

    assert len(keys) == len(
        out_keys
    ), "Number of keys does not match number of output keys"

    out = {}
    with h5py.File(feats_path, "r") as open_file:
        if verbose:
            keys_to_print = open_file[group].keys()
            print("Keys in dataset:")
            pprint.pprint(keys_to_print)

        for in_key, out_key in zip(keys, out_keys):
            out[out_key] = open_file[group][in_key][:]
            if verbose:
                print(f"Extracted {out_key, out[out_key].shape}:")

    return out
