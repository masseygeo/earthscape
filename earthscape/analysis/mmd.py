
import os
import glob
import numpy as np
import pandas as pd
import rasterio
from hyppo.ksample import MMD



def select_samples(dirs_list, size, rng, patch_id_file_suffix='areas.csv'):
    """
    Randomly select a subset of patch identifiers from one or more directories.

    Parameters
    ----------
    dirs_list : sequence of str or os.PathLike
        Directories containing files with patch identifiers.
    size : int
        Number of samples to select.
    rng : int or numpy.random.RandomState or numpy.random.Generator
        Random seed or random number generator used for sampling.
    patch_id_file_suffix : str, optional
        Suffix used to identify files containing patch identifiers.

    Returns
    -------
    list of str
        Randomly selected patch identifiers.
    """
    df_list = []
    for d in dirs_list:
        path = glob.glob(os.path.join(d, f"*{patch_id_file_suffix}"))[0]
        df = pd.read_csv(path)
        df_list.append(df)
    df = pd.concat(df_list)
    ids = df['patch_id'].sample(n=size, replace=False, random_state=rng).to_list()

    return ids



def _raster_percentiles(path, percentiles):
    """
    Compute per-band intensity percentiles from a raster.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the raster file.
    percentiles : sequence of float
        Percentiles to compute (e.g., [5, 50, 95]).

    Returns
    -------
    numpy.ndarray
        Concatenated percentile values for each band with shape
        [num_bands * len(percentiles)].
    """
    features = []
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)

    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    features = []
    for band in data:
        features.append(np.percentile(band, percentiles))

    return np.concatenate(features)



def _patch_sample_matrix(dirs, patch_ids, in_features, percentiles):
    """
    Construct a feature matrix by sampling raster patches and computing percentiles.

    Parameters
    ----------
    dirs : sequence of str or os.PathLike
        Directories searched for raster files.
    patch_ids : sequence of str
        Patch identifiers used to locate raster files.
    in_features : sequence of str
        Feature names appended to each patch identifier to form filenames.
    percentiles : sequence of float
        Percentiles to compute for each raster band.

    Returns
    -------
    numpy.ndarray
        Array of shape [N, F], where N is the number of patches and F is the
        total number of computed features across all inputs and percentiles.
    """
    
    patches = []

    for pid in patch_ids:

        features = []

        for feat in in_features:

            for d in dirs:

                path = os.path.join(d, f"{pid}_{feat}")

                if os.path.isfile(path):
                    break

            features.append(_raster_percentiles(path, percentiles))

        patches.append(np.concatenate(features))

    return np.vstack(patches)



def _zscore_pair(X_a, X_b):
    """
    Apply z-score normalization to two datasets using shared statistics.

    Parameters
    ----------
    X_a : numpy.ndarray
        First array with shape [N_a, F].
    X_b : numpy.ndarray
        Second array with shape [N_b, F].

    Returns
    -------
    tuple of numpy.ndarray
        Normalized arrays (X_a, X_b) using mean and standard deviation
        computed over the combined data.
    """

    X_all = np.vstack([X_a, X_b])

    mu = X_all.mean(axis=0)
    sd = X_all.std(axis=0)

    sd[sd == 0] = 1

    X_a = (X_a - mu) / sd
    X_b = (X_b - mu) / sd

    return X_a, X_b


def _median_gamma(X):
    """
    Estimate an RBF kernel gamma parameter using the median heuristic.

    Parameters
    ----------
    X : numpy.ndarray
        Input array with shape [N, F].

    Returns
    -------
    float
        Gamma value defined as 1 / (2 * median pairwise squared distance),
        excluding zero distances.
    """

    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    vals = d2[np.triu_indices_from(d2, k=1)]
    vals = vals[vals > 0]
    med = np.median(vals)

    return 1.0 / (2.0 * med)



def compute_mmd(dirs_a, dirs_b, patch_ids_a, patch_ids_b, in_features, gamma=None, percentiles=(10,25,50,75,90)):
    """
    Compute Maximum Mean Discrepancy (MMD) between two datasets of raster patches.

    Parameters
    ----------
    dirs_a : sequence of str or os.PathLike
        Directories for dataset A.
    dirs_b : sequence of str or os.PathLike
        Directories for dataset B.
    patch_ids_a : sequence of str
        Patch identifiers for dataset A.
    patch_ids_b : sequence of str
        Patch identifiers for dataset B.
    in_features : sequence of str
        Feature names used to locate raster files.
    gamma : float or None, optional
        RBF kernel parameter. If None, estimated using the median heuristic.
    percentiles : sequence of float, optional
        Percentiles computed per raster band.

    Returns
    -------
    stat : float
        MMD test statistic.
    p : float
        p-value from the permutation test.
    gamma_new : float
        Gamma value estimated from the combined data using the median heuristic.
    """

    X_a = _patch_sample_matrix(dirs_a, patch_ids_a, in_features, percentiles)
    X_b = _patch_sample_matrix(dirs_b, patch_ids_b, in_features, percentiles)

    X_a, X_b = _zscore_pair(X_a, X_b)

    X_all = np.vstack([X_a, X_b])
    gamma_new = _median_gamma(X_all)

    if gamma is None:
        stat, p = MMD(compute_kernel='rbf', bias=False, gamma=gamma_new).test(X_a, X_b, reps=1000)

    else:
        stat, p = MMD(compute_kernel='rbf', bias=False, gamma=gamma).test(X_a, X_b, reps=1000)

    return stat, p, gamma_new
