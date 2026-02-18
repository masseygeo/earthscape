
import os
import rasterio
import numpy as np
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

    data_dirs : sequence of str or os.PathLike
        Directories containing "{patch_id}_labels.csv" and associated channel files (e.g., "{patch_id}_dem.tif").

    modalities : dict
        Mapping of informal modality name to input feature channel path suffixes, channel mean, channel standard deviations. 
        Expected format:
        
        {
        "informal modality name": 
            {
            "channels": [sequence of str], 
            "mean": [sequence of float or None], 
            "sd": [sequence of float or None]
            }
        }

    normalize : bool, default=True
        Apply per-channel normalization.

    augment : bool, default=False
        Apply random horizontal/vertical flips and 90 degree rotations
        consistently across modalities.
    """

    def __init__(self, patch_ids, data_dirs, modalities, normalize=True, augment=False):
        self.ids = list(patch_ids)
        self.data_dirs = list(data_dirs)
        self.modalities = modalities
        self.normalize = normalize
        self.augment = augment
        
        # dict index for labels & modality paths for each patch -> {patch ID: {'label': torch.Tensor, 'modality_paths': sequence of str}}
        self._index = self._build_index()

        # dict for normalization using mean & standard deviation tensors for each modality
        self._norm = self._build_normalizers()


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):
        ##### get patch information...
        patch_id = self.ids[idx]          # unique patch ID
        entry = self._index[patch_id]     # label & modality paths for patch

        ##### get label tensor for patch...
        data = {'label': entry['label']}

        ##### get image tensor for patch...
        for mod, paths in entry['modality_paths'].items():

            # stack modality channels & return tensor of shape (C, H, W)
            t = self.stack_images(paths)

            # fill background NaN in categorical images to 0
            # NOTE: should only affect osm, nhd, RGB+NIR, & mask
            t = torch.nan_to_num(t, nan=0.0)
            
            # normalize per channel (optional)...
            norm = self._norm[mod]
            if norm is not None:
                mean, sd = norm
                t = (t - mean) / (sd + 1e-8)

            # final modality tenor
            data[mod] = t

        ##### apply random augmentations (optional)...
        if self.augment:
            params = self._sample_aug_params()
            data = self._apply_aug(data, *params)

        return data


    def _get_patch_dir_and_label(self, patch_id):
        """
        Resolve correct directory for `patch_id` and load its label tensor. 
        """
        for resolved_dir in self.data_dirs:
            label_path = os.path.join(resolved_dir, f"{patch_id}_labels.csv")
            if os.path.isfile(label_path):
                label = np.loadtxt(label_path)
                label = torch.from_numpy(label).type(torch.float32)
                return resolved_dir, label


    def _get_modality_paths(self, patch_id, resolved_dir):
        """
        Build per-modality list of channel file paths for a given patch.
        """
        modality_paths = {}
        for mod_name, data in self.modalities.items():
            modality_paths[mod_name] = [os.path.join(resolved_dir, f"{patch_id}_{ext}") for ext in data['channels']]
        return modality_paths
    

    def _build_index(self):
        """
        Construct dict mapping of {`patch_id`: {`label`: torch.Tensor, `modality_paths`: sequence of str}}.
        """
        index = {}
        for patch_id in self.ids:
            resolved_dir, label = self._get_patch_dir_and_label(patch_id)
            modality_paths = self._get_modality_paths(patch_id, resolved_dir)
            index[patch_id] = {'label': label, 'modality_paths': modality_paths}
        return index


    def _build_normalizers(self):
        """
        Build (mean, sd) tensors shaped (C,1,1) per modality for continuous input features. Binary 
        or categorical input feautres are treated as identity normalization.
        """
        norm = {}
        for mod_name, data in self.modalities.items():
            means = data.get('mean')
            sds = data.get('sd')

            if self.normalize and means is not None:
                mean_vals = [0.0 if m is None else float(m) for m in means]
                sd_vals = [1.0 if s is None else float(s) for s in sds]

                mean = torch.tensor(mean_vals, dtype=torch.float32)[:, None, None]
                sd = torch.tensor(sd_vals, dtype=torch.float32)[:, None, None]
                norm[mod_name] = (mean, sd)
            else:
                norm[mod_name] = None
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
        for mod in self.modalities.keys():
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
    