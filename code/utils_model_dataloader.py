


import os
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import normalize




def randomly_select_indpendent_patch_sets(gdf_patches, val_size, seed=111):

  # select random patches...
  rng = np.random.default_rng(seed=seed)                                          # create random range with seed
  random_idx = rng.choice(gdf_patches.index, size=val_size, replace=False)        # choose random patches
  gdf_select = gdf_patches.loc[random_idx].copy()                                 # isolate selected patches

  # remove selected patches & overlapping patches for spatially independent sets of patches...
  gdf_remaining = gdf_patches[~gdf_patches.index.isin(gdf_select.index)].copy()              # remove selected patches from gdf
  overlapping_patches = gdf_remaining.sjoin(gdf_select, how='inner', predicate='overlaps')   # identify patches that overlap selected patches
  gdf_remaining = gdf_remaining[~gdf_remaining.index.isin(overlapping_patches.index)]        # remove overlapping patches from gdf

  # reset index and return two gdf's 
  gdf_select.reset_index(drop=True, inplace=True)
  gdf_remaining.reset_index(drop=True, inplace=True)

  return gdf_select, gdf_remaining




class MultiModalDataset(Dataset):
  def __init__(self, ids, data_dir, modalities, norm_params=None, augment=False, task='classification'):
    self.ids = ids                   # list of patch IDs
    self.data_dir = data_dir         # directory containing all data
    self.modalities = modalities     # dictionary of modalities (modality name : path extension)
    self.norm_params = norm_params   # boolean; normalize modalities
    self.augment = augment           # bool; augment modalities - random horizontal flip, vertical flip, & 90 degree rotations
    self.task = task                 # type of problem - classification or segmentation

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, idx):

    ##### Patch id
    patch_id = self.ids[idx]

    ##### Labels
    if self.task == 'classification':
      label_path = os.path.join(self.data_dir, f"{patch_id}_labels.csv")
      label = np.loadtxt(label_path)
      label = torch.from_numpy(label).unsqueeze(0)
      label = label.type(torch.float)
    
    data = {'label': label}

    ##### Modalities
    for modality, channel_paths in self.modalities.items():
      paths = [os.path.join(self.data_dir, f"{patch_id}_{file_extension}") for file_extension in channel_paths]
      image = self.stack_images(paths)
      
      if self.norm_params:
        if modality in self.norm_params.keys():
          image = normalize(image, self.norm_params[modality][0], self.norm_params[modality][1])
      
      data[modality] = image

    ##### Apply random augmentation(s)
    if self.augment:
        
        if np.random.uniform(low=0, high=1) > 0.5:
          for modality in self.modalities.keys():
            data[modality] = v2.functional.horizontal_flip(data[modality])
    
        if np.random.uniform(low=0, high=1) > 0.5:
          for modality in self.modalities.keys():
            data[modality] = v2.functional.vertical_flip(data[modality])
        
        angle = np.random.choice([0, 90, 180, 270])
        for modality in self.modalities.keys():
            data[modality] = v2.functional.rotate(data[modality], angle=angle)

    return data

  @staticmethod
  def stack_images(paths_list):
    """
    Function to extract image arrays, stack if multiple images provided, and return tensor with shape [Channels, Height, Width].
    """
    # initialize list to hold image arrays
    src_arrays = []

    # iterate through image paths
    for path in paths_list:

      # open image
      with rasterio.open(path) as src:
        data = src.read(1)                       # read channel 1 as array (all input should be 1 channel)
        src_arrays.append(data)                  # append array to list
    image_array = np.stack(src_arrays, axis=0)   # stack image arrays along channel dimension
    return torch.from_numpy(image_array)         # return tensor with shape [channels, h, w]






def prep_image_for_plot(batch_image):
  """Function to prepare image tensor from DataLoader batch for visualization."""
  image = batch_image.clone().detach().numpy()
  min_val = image.min()
  max_val = image.max()
  image = (image - min_val) / (max_val-min_val)
  image = np.transpose(image, (1, 2, 0))
  return image





def get_norm_data(image_paths):
  """
  Function to calculate mean and standard deviation of 1-channel images.
  """
  total_sum = 0
  total_sum_squares = 0
  total_pixels = 0

  for path in image_paths:
    with rasterio.open(path) as src:
      data = src.read(1, masked=True)             # should not be any masked values, but just in case
      data = data.compressed()                    # this will remove any masked nodata values (if any)

      total_sum += np.sum(data)
      total_sum_squares += np.sum(data**2)
      total_pixels += data.size
      
  mean = total_sum / total_pixels
  var = (total_sum_squares / total_pixels) - mean**2
  sd = np.sqrt(var)
  return mean, sd

