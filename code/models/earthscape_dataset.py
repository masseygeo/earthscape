import os
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F



class EarthScape_Dataset(Dataset):
    def __init__(self, patch_ids, data_dirs, modalities, normalize=True, augment=False, resize=False):
        self.ids = patch_ids           # list of patch IDs
        self.data_dirs = data_dirs     # list of directories containing patches 
        self.normalize = normalize     # normalize continuous images
        self.augment = augment         # apply random horizontal & veritcal flips + 90 deg rotations
        self.resize = resize           # resize image to 224x224 - for ViT
        
        # nested dict of modality name & file extensions + mean + standard deviation
        self.modalities = modalities    
        # Example: {'ep' : {
        #                'extensions': [ep_5x5.tif, ep_11x11.tif], 
        #                'mean': [15.2, 16.23], 
        #                'sd': [12.3, 14.2]}

        # get correct patch dir, label path, & paths for all modality channels...
        self._index = {}
        
        # iterate through each patch...
        for pid in self.ids:
            
            # iterate through each data dir to get correct directory & label path...
            resolved_dir = None
            label_path = None
            for d in self.data_dirs:
                candidate_path = os.path.join(d, f"{pid}_labels.csv")
                if os.path.isfile(candidate_path):
                    resolved_dir = d
                    label_path = candidate_path
                    break
            
            # iterate through each nested modality dict...
            modality_paths = {}
            for name, data in self.modalities.items():
                
                # iterate through path extensions & get modality channel paths...
                modality_paths[name] = []
                for ext in data['extensions']:
                    mod_paths = os.path.join(resolved_dir, f"{pid}_{ext}")
                    modality_paths[name].append(mod_paths)
            
            self._index[pid] = {'dir': resolved_dir, 
                                'label_path': label_path, 
                                'modality_paths': modality_paths}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        ##### get patch information...
        patch_id = self.ids[idx]          # unique patch ID
        entry = self._index[patch_id]     # data dir, label path, & modality paths

        ##### get label tensor...
        label = np.loadtxt(entry['label_path'])
        label = torch.from_numpy(label).type(torch.float32)
        data = {'label': label}

        ##### get stacked & normalized image tensor...
        for mod, paths in entry['modality_paths'].items():

            # stack modality channels & return tensor - [C, H, W]
            t = self.stack_images(paths)

            # normalize each channel (optional; continuous/non-categorical only)...
            if self.normalize & (self.modalities[mod]['mean'] != None):
                mean = self.modalities[mod]['mean']
                sd = self.modalities[mod]['sd']
                mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
                sd = torch.tensor(sd, dtype=torch.float32)[:, None, None]
                t = (t - mean) / (sd + 1e-8)
            
            # add stacked & normalized image tensor to Dataset
            data[mod] = t

        ##### apply random augmentation(s)...
        if self.augment:
            if torch.rand(()) > 0.5:
                for mod in self.modalities.keys():
                    data[mod] = torch.flip(data[mod], dims=[2])
        
            if torch.rand(()) > 0.5:
                for mod in self.modalities.keys():
                    data[mod] = torch.flip(data[mod], dims=[1])
            
            k = torch.randint(low=0, high=4, size=(1,)).item()
            for mod in self.modalities.keys():
                data[mod] = torch.rot90(data[mod], k=k, dims=(1,2))

        # ##### resize image to 224x224 (optional)...
        # if self.resize:
        #     resized = img_tensor.unsqueeze(0)              
        #     img_tensor = F.interpolate(img_tensor, size=(224, 224), mode='bicubic', align_corners=False)    # [1, C, 224, 224]
        #     img_tensor = img_tensor.squeeze(0)                                          # remove batch dim - [C, 224, 224]
        # data[modality] = img_tensor

        return data

    @staticmethod
    def stack_images(paths_list):
        """
        Function to extract image arrays, stack if multiple images provided, and return tensor with shape [Channels, Height, Width].
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