from utils_models import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd

def plot_patch_counts(patch_id_list, patch_label_path):
  df = pd.read_csv(patch_label_path)
  df_subset = df.loc[df['patch_id'].isin(patch_id_list)]
  counts = df_subset.iloc[:, 1:].sum(axis=0)
  return counts


def plot_patch_areas(patch_id_list, patch_area_path):
  df = pd.read_csv(patch_area_path)
  df_subset = df.loc[df['patch_id'].isin(patch_id_list)]
  areas = df_subset.apply(lambda x: x*256*256)
  areas = df_subset.sum(axis=0) / len(df_subset)
  return areas



# count_path = r'../data/warren/patches_256_50_labels.csv'

# df = pd.read_csv(count_path)


# counts = plot_patch_counts(random_patches, count_path)
# counts = pd.DataFrame(counts)
# sns.barplot(data=counts, x=counts.index, y=0)

area_path = r'../data/warren/patches_256_50_areas.csv'

df = pd.read_csv(area_path)

random_patches = np.random.choice(df['patch_id'], size=500, replace=False)

df_long = df.iloc[:, 1:].melt(var_name='Geologic Map Unit', value_name='Proportion')


sns.boxplot(data=df_long, x='Geologic Map Unit', y='Proportion', showfliers=False, fill=False, color='k', width=0.5, linewidth=1)

sns.stripplot(data=df_long, x='Geologic Map Unit', y='Proportion', jitter=True, edgecolor='k', linewidth=0.2, alpha=0.005, facecolor='#3A6D8C')

# sns.violinplot(data=df_long, x='Geologic Map Unit', y='Proportion', inner=None, linewidth=0.5, color='blue')

# plt.yscale('log')
plt.ylim(0,1)