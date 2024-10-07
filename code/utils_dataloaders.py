import os
import rasterio
import geopandas as gpd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2



def randomly_select_indpendent_patch_sets(patches_path, test_size=0.1, val_size=0.1, seed=111):
  
  # get input data...
  gdf = gpd.read_file(patches_path)            # read patches into geodataframe
  test_size = int(test_size * len(gdf))        # get size of test set
  val_size = int(val_size * len(gdf))          # get size of validation set
  rng = np.random.default_rng(seed=seed)       # create random range with seed

  # get random test set of patches...
  random_test_idx = rng.choice(gdf.index, size=test_size, replace=False)
  gdf_test = gdf.loc[random_test_idx].copy()
  # gdf_test.reset_index(drop=True, inplace=True)


  # spatial join to exclude patches intersecting test set
  # gdf = gpd.overlay(gdf, gdf_test, how='difference')           
  intersecting_patches = gpd.sjoin(gdf, gdf_test, how='inner', predicate='overlaps')
  gdf = gdf[~gdf.index.isin(intersecting_patches.index)]
  gdf.reset_index(drop=True, inplace=True)


  return gdf, gdf_test




  



class MultiModalDataset(Dataset):
  def __init__(self, ids, data_dir, transform_rgb=None, transform_dem=None, horiz_flip=False, vert_flip=False, rand_rot=False):
    self.ids = ids                               # list of patch IDs
    self.data_dir = data_dir                     # directory containing all data
    self.transform_rgb = transform_rgb           # transform for aerial rgb
    self.transform_dem = transform_dem           # transform for dem
    self.horiz_flip = horiz_flip
    self.vert_flip = vert_flip
    self.rand_rot = rand_rot


  def __len__(self):
    return len(self.ids)

  def __getitem__(self, idx):
    unique_id = self.ids[idx]

    ##### Label vector
    label_path = os.path.join(self.data_dir, f"{unique_id}_labels.csv")
    label = np.loadtxt(label_path)                                    # read label as array
    label = torch.from_numpy(label).unsqueeze(0)                      # create tensor of size [1, 7]
    label = label.type(torch.float)
    
    ##### Aerial (RGB) image
    r_path = os.path.join(self.data_dir, f"{unique_id}_aerialr.tif")
    g_path = os.path.join(self.data_dir, f"{unique_id}_aerialg.tif")
    b_path = os.path.join(self.data_dir, f"{unique_id}_aerialb.tif")
    rgb_image = self.stack_images([r_path, g_path, b_path])           # create tensor of size [3, h, w]
    if self.transform_rgb:
      rgb_image = self.transform_rgb(rgb_image)                       # apply transform if provided

    ##### DEM image
    dem_path = os.path.join(self.data_dir, f"{unique_id}_dem.tif")
    dem_image = self.stack_images([dem_path])                         # create tensor of size [1, h, w]
    if self.transform_dem:
      dem_image = self.transform_dem(dem_image)                       # apply transform if provided

    ##### Apply random augmentation(s)
    if self.horiz_flip:
      if np.random.uniform(low=0, high=1) > 0.5:
        rgb_image = v2.functional.horizontal_flip(rgb_image)
        dem_image = v2.functional.horizontal_flip(dem_image)
    if self.vert_flip:
      if np.random.uniform(low=0, high=1) > 0.5:
        rgb_image = v2.functional.vertical_flip(rgb_image)
        dem_image = v2.functional.vertical_flip(dem_image)
    if self.rand_rot:
        angle = np.random.choice([0, 90, 180, 270])
        rgb_image = v2.functional.rotate(rgb_image, angle=angle)
        dem_image = v2.functional.rotate(dem_image, angle=angle)

    return {'rgb': rgb_image, 'dem': dem_image, 'label': label}

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