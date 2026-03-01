
import os
import glob
import rasterio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset




class ESDataset_Classification(Dataset):
    """
    Patch-based multi-modal classification dataset.

    Each sample is a dict with:
        - `label`: `torch.Tensor` of shape (K,)
        - one key per input modality feature: `torch.Tensor` of shape (C, H, W)

    Parameters
    ----------
    patch_ids : sequence of str
        Patch identifiers.

    patch_dirs : sequence of str or os.PathLike
        Directories containing image files for sample patches in `patch_ids`.

    input_features : dict
        Mapping of informal input feature name to channel path suffixes, channel mean, channel standard deviations. 
        Expected format:
        
        {
        "informal name": 
            {
            "channels": [sequence of str], 
            "mean": [sequence of float or None], 
            "sd": [sequence of float or None]
            }
        }

    areas_path : str or os.PathLike
        Path to CSV containing patch ID and associated class-area 
        proportions in the assocated patch.

    label_threshold : float or None, default=None
        Apply minimum class-area threshold for one-hot labels (float 0.0 - 1.0), where 
        class presence is defined as area-proportion greater than given threshold. If
        `None` is provided, the raw class-area proportions are used as targets, enabling 
        regression or multilabel soft labels.

    normalize : bool, default=True
        Apply per-channel normalization.

    augment : bool, default=False
        Apply random horizontal/vertical flips and 90 degree rotations
        consistently across modalities.
    """

    def __init__(self, patch_ids, patch_dirs, input_features, areas_path, label_threshold=None, normalize=True, augment=False):
        self.ids = list(patch_ids)
        self.patch_dirs = list(patch_dirs)
        self.input_features = input_features
        # self.channels = input_features['channels']
        self.areas_path = areas_path
        self.label_threshold = label_threshold
        self.normalize = normalize
        self.augment = augment

        # keep only patch_ids that exist in the provided patch_dirs (really only for smoke set tests)
        self.ids = [pid for pid in self.ids if self._resolve_dir(pid) is not None]

        # index dict of {patch_id: {'areas': tensor of shape (K,), 'input_paths': sequence of str or os.PathLike}}
        self._index = self._build_input_index()

        # normalization dict of mean & standard deviation tensors for each input
        self._norm = self._build_normalizers()


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):
        
        patch_id = self.ids[idx]          # unique patch ID
        entry = self._index[patch_id]     # label & modality paths for patch
        data = {}                         # initialize dict for return

        ##### get label tensor for patch...
        # if labels are provided (supervised) (if not, then image tensor returned only without label)
        if self.areas_path is not None:
            if self.label_threshold is None:                                        # use class-area proportions as targets
                label = entry['areas'].to(torch.float32)
            else:                                                                   # use one-hot labels as targets
                label = (entry['areas'] > self.label_threshold).to(torch.float32)
            data['label'] = label


        ##### get image tensor for patch...
        for name, paths in entry['input_paths'].items():

            # stack modality channels & return tensor of shape (C, H, W)
            t = self.stack_images(paths)
            
            # normalize per channel (optional)...
            norm = self._norm[name]
            if norm is not None:
                mean, sd = norm
                t = (t - mean) / (sd + 1e-8)

            # final modality tenor
            data[name] = t

        ##### apply random augmentations (optional)...
        if self.augment:
            params = self._sample_aug_params()
            data = self._apply_aug(data, *params)

        return data


    def _build_input_index(self):
        """
        Build lookup index for patch sample areas and input channel paths. Dict of:

        {patch_id: {'areas': torch.Tensor of shape (K,), 'input_paths': list of channel paths}} 
        """
        index = {}
        areas = self._build_areas_index() if self.areas_path is not None else None
        for patch_id in self.ids:
            resolved_dir = self._resolve_dir(patch_id)
            input_paths = self._get_input_paths(patch_id, resolved_dir)

            # supervised with labels...
            if self.areas_path is not None:
                class_areas = areas[patch_id]
                index[patch_id] = {'areas': class_areas, 'input_paths': input_paths}
            
            # prediction of unknowns (no labels)...
            else:
                index[patch_id] = {'input_paths': input_paths}
        
        return index


    def _build_areas_index(self):
        """Build dict of patch_id and associated label class-area proportions. Dict of:
        
        {'patch_id': torch.Tensor of shape (K,)}
        
        """
        index = {}
        df = pd.read_csv(self.areas_path)
        df = df.loc[df['patch_id'].isin(self.ids)]
        df.set_index('patch_id', inplace=True, drop=True)
        for pid, areas in df.iterrows():
            index[pid] = torch.from_numpy(areas.to_numpy()).to(dtype=torch.float32)
        return index
    

    def _resolve_dir(self, patch_id):
        """
        Resolve correct data directory for specific patch.
        """
        first_mod = next(iter(self.input_features.values()))
        sentinel_ext = first_mod["channels"][0]
        fname = f"{patch_id}_{sentinel_ext}"
        for resolved_dir in self.patch_dirs:
            if os.path.exists(os.path.join(resolved_dir, fname)):
                return resolved_dir
            # paths = glob.glob(os.path.join(resolved_dir, f"{patch_id}_*"))
            # if len(paths) > 0:
            #     return resolved_dir


    def _get_input_paths(self, patch_id, resolved_dir):
        """
        Build per-modality list of channel file paths for a given patch.
        """
        index = {}
        for name, data in self.input_features.items():
            index[name] = [os.path.join(resolved_dir, f"{patch_id}_{ext}") for ext in data['channels']]
        return index
    

    def _build_normalizers(self):
        """
        Build (mean, sd) tensors shaped (C,1,1) per modality for continuous input features. Binary or categorical input feautres are treated as identity normalization.
        """
        norm = {}
        for name, data in self.input_features.items():
            means = data.get('mean')
            sds = data.get('sd')

            if self.normalize and means is not None:
                mean_vals = [0.0 if m is None else float(m) for m in means]
                sd_vals = [1.0 if s is None else float(s) for s in sds]

                mean = torch.tensor(mean_vals, dtype=torch.float32)[:, None, None]
                sd = torch.tensor(sd_vals, dtype=torch.float32)[:, None, None]
                norm[name] = (mean, sd)
            else:
                norm[name] = None
        return norm
    

    def _sample_aug_params(self):
        """
        Create random augmentation parameters for horizontal flips, vertical flips, and 90 degree rotations.
        """
        hflip = torch.rand(()) > 0.5
        vflip = torch.rand(()) > 0.5
        k = int(torch.randint(low=0, high=4, size=(1,)).item())
        return hflip, vflip, k
    

    def _apply_aug(self, data, hflip, vflip, k):
        """
        Apply spatial transforms consistently across all modalities (not label).
        """
        for mod in self.input_features.keys():
            x = data[mod]
            if hflip:
                x = torch.flip(x, dims=[2])  # W
            if vflip:
                x = torch.flip(x, dims=[1])  # H
            if k:
                x = torch.rot90(x, k=k, dims=(1, 2))
            data[mod] = x
        return data


    @staticmethod
    def stack_images(paths_list):
        """
        Load single-band GeoTIFFs and stack into (C,H,W) float32 torch.Tensor.
        """
        # initialize list to hold image arrays
        src_arrays = []

        # extract image arrays from GeoTIFF images...
        for path in paths_list:
            with rasterio.open(path) as src:    # open GeoTIFF
                data = src.read(1)              # read channel 1 as array (all input should be 1 channel)
                src_arrays.append(data)         # append array to list
        
        # stack image arrays along channel dimension
        stacked = np.stack(src_arrays, axis=0)

        # return tensor with shape [C, H, W]
        return torch.from_numpy(stacked).to(torch.float32).contiguous()
    