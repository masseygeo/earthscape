
import pandas as pd

def get_norm_stats(stats_path, modality_configs):
    """
    Compute per-channel normalization statistics for each modality.

    Parameters
    ----------
    stats_path : str or pathlib.Path
        Path to a CSV file containing training-set statistics. The first column
        contains channel identifiers and the CSV includes ``mean`` and ``sd`` columns.
    modality_configs : dict
        Dictionary of modality configurations. Each value must contain a
        ``'channels'`` list specifying channel identifiers.

    Returns
    -------
    dict
        The same ``modality_configs`` object, modified in-place. Each modality is
        extended with ``'mean'`` and ``'sd'`` lists aligned with ``'channels'``.
        Channels that should not be normalized have ``None`` entries.
    """

    # read stats CSV to df
    df = pd.read_csv(stats_path)

    # iterate through values in modality_configs dictionary
    for _, data in modality_configs.items():

        # add two additional values for mean and sd
        data.update({'mean': [], 'sd': []})
        
        # iterate through channels in dictionary list named 'channels' containing modality file suffixes.
        for c in data['channels']:
            
            # categorical images should not have normalization stats (0 or 1)
            if ('osm' in c) or ('nhd' in c) or ('mask' in c):
                data['mean'].append(None)
                data['sd'].append(None)
            
            # other images should have normalization stats from training dataset
            else:
                row = df.loc[df[df.columns[0]] == c]
                data['mean'].append(row['mean'].item())
                data['sd'].append(row['sd'].item())
    
    # return modified dictionary
    return modality_configs