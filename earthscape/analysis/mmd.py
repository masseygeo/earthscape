
import os
import glob
import numpy as np
import pandas as pd
import rasterio
from hyppo.ksample import MMD



def select_samples(dirs_list, size, rng, patch_id_file_suffix='areas.csv'):
    df_list = []
    for d in dirs_list:
        path = glob.glob(os.path.join(d, f"*{patch_id_file_suffix}"))[0]
        df = pd.read_csv(path)
        df_list.append(df)
    df = pd.concat(df_list)
    ids = df['patch_id'].sample(n=size, replace=False, random_state=rng).to_list()

    return ids



def _raster_percentiles(path, percentiles):
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

    X_all = np.vstack([X_a, X_b])

    mu = X_all.mean(axis=0)
    sd = X_all.std(axis=0)

    sd[sd == 0] = 1

    X_a = (X_a - mu) / sd
    X_b = (X_b - mu) / sd

    return X_a, X_b


def _median_gamma(X):

    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    vals = d2[np.triu_indices_from(d2, k=1)]
    vals = vals[vals > 0]
    med = np.median(vals)

    return 1.0 / (2.0 * med)



def compute_mmd(dirs_a, dirs_b, patch_ids_a, patch_ids_b, in_features, gamma=None, percentiles=(10,25,50,75,90)):

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








##################
###### OLD ######
##################

# def collect_patch_features(dirs, glob_patterns, percentiles=(10, 25, 50, 75, 90)):

#     # Normalize inputs
#     if isinstance(dirs, str):
#         dirs = [dirs]
#     if isinstance(glob_patterns, str):
#         glob_patterns = [glob_patterns]

#     # Gather files for each pattern
#     pattern_to_files = {pat: [] for pat in glob_patterns}
#     for d in dirs:
#         for pat in glob_patterns:
#             pattern = os.path.join(d, pat)
#             pattern_to_files[pat].extend(glob.glob(pattern))

#     # Build mapping: patch_id -> {pattern: filepath}
#     patch_dict = {}
#     for pat, files in pattern_to_files.items():
#         # Suffix after '*' (assumes a single '*' in the pattern)
#         if "*" in pat:
#             suffix = pat.split("*", 1)[1]
#         else:
#             # No '*': treat whole pattern as suffix after path
#             suffix = os.path.basename(pat)

#         for f in files:
#             base = os.path.basename(f)
#             if suffix and base.endswith(suffix):
#                 patch_id = base[:-len(suffix)]
#             else:
#                 # Fallback: use basename without extension
#                 patch_id = os.path.splitext(base)[0]

#             if patch_id not in patch_dict:
#                 patch_dict[patch_id] = {}
#             patch_dict[patch_id][pat] = f

#     # Keep only patches that have ALL patterns
#     valid_patches = [
#         (pid, patch_dict[pid])
#         for pid in patch_dict
#         if all(pat in patch_dict[pid] for pat in glob_patterns)
#     ]

#     if len(valid_patches) == 0:
#         raise ValueError(
#             f"No patches found that contain all patterns: {glob_patterns}"
#         )

#     feature_list = []

#     # Process each patch
#     for patch_id, file_map in valid_patches:
#         patch_feats = []

#         # Iterate patterns in the given order so channel ordering is deterministic
#         for pat in glob_patterns:
#             fpath = file_map[pat]
#             with rasterio.open(fpath) as src:
#                 arr = src.read()  # (C, H, W) or (H, W)
#                 nodata = src.nodata

#             # Ensure (C, H, W)
#             if arr.ndim == 2:
#                 arr = arr[np.newaxis, :, :]

#             C = arr.shape[0]
#             for c in range(C):
#                 band = arr[c].astype(np.float32)

#                 # Handle nodata
#                 if nodata is not None:
#                     band = np.where(band == nodata, np.nan, band)

#                 vals = band.reshape(-1)
#                 vals = vals[~np.isnan(vals)]

#                 if vals.size == 0:
#                     p = np.zeros(len(percentiles), dtype=np.float32)
#                 else:
#                     p = np.percentile(vals, percentiles).astype(np.float32)

#                 patch_feats.append(p)

#         feature_list.append(np.concatenate(patch_feats))

#     return np.vstack(feature_list)


# def gaussian_kernel_matrix(X, Y, gamma):

#     X_norm = np.sum(X ** 2, axis=1)[:, None]
#     Y_norm = np.sum(Y ** 2, axis=1)[None, :]
#     dist_sq = X_norm + Y_norm - 2 * X.dot(Y.T)
#     return np.exp(-gamma * dist_sq)


# def median_heuristic_gamma(X):

#     n = X.shape[0]
#     if n > 2000:
#         idx = np.random.choice(n, size=2000, replace=False)
#         X_sub = X[idx]
#     else:
#         X_sub = X

#     X_norm = np.sum(X_sub ** 2, axis=1)
#     dist_sq = X_norm[:, None] + X_norm[None, :] - 2 * X_sub.dot(X_sub.T)
#     i_upper = np.triu_indices_from(dist_sq, k=1)
#     vals = dist_sq[i_upper]
#     vals = vals[vals > 0]

#     if vals.size == 0:
#         return 1.0

#     med = np.median(vals)
#     if med <= 0:
#         return 1.0

#     return 1.0 / (2.0 * med)


# def mmd_rbf(X, Y, gamma=None):

#     if gamma is None:
#         pooled = np.vstack([X, Y])
#         gamma = median_heuristic_gamma(pooled)

#     Kxx = gaussian_kernel_matrix(X, X, gamma)
#     Kyy = gaussian_kernel_matrix(Y, Y, gamma)
#     Kxy = gaussian_kernel_matrix(X, Y, gamma)

#     n_x = X.shape[0]
#     n_y = Y.shape[0]

#     mmd2 = (Kxx.sum() / (n_x ** 2) + Kyy.sum() / (n_y ** 2) - 2.0 * Kxy.sum() / (n_x * n_y))
    
#     return float(mmd2), gamma



# def compute_mmd_from_dirs(dirs_region_a, dirs_region_b, glob_patterns, percentiles=(10, 25, 50, 75, 90), gamma=None):

#     X_A = collect_patch_features(dirs_region_a, glob_patterns, percentiles)
#     X_B = collect_patch_features(dirs_region_b, glob_patterns, percentiles)

#     # Min–max scale using pooled stats
#     X_all = np.vstack([X_A, X_B])
#     mins = X_all.min(axis=0)
#     maxs = X_all.max(axis=0)
#     denom = maxs - mins
#     denom[denom == 0] = 1.0

#     X_A_scaled = (X_A - mins) / denom
#     X_B_scaled = (X_B - mins) / denom

#     return mmd_rbf(X_A_scaled, X_B_scaled, gamma=gamma)