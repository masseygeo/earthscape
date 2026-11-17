
import numpy as np
import pandas as pd
# import re




def parse_input_label(x):
    """
    Parse an input label into family and spatial support.

    Rules:
    1. If input contains '+':
       family = 'multimodal', spatial support = 'multi-scale'
    2. If input ends with '-ms' and does not contain '+':
       family = 'multi-scale', spatial support = 'multi-scale'
    3. If input has no '-':
       family = 'raw', spatial support = 5
    4. Otherwise split on '-':
       family = first part
       spatial support derived from second part
    """
    x = str(x)

    if "+" in x:
        return pd.Series({
            "family": "multimodal",
            "spatial support": "multi-scale"
        })

    if x.endswith("-ms"):
        return pd.Series({
            "family": "multi-scale",
            "spatial support": "multi-scale"
        })

    if "-" not in x:
        return pd.Series({
            "family": "raw",
            "spatial support": 5
        })

    family, support_code = x.split("-", 1)
    support_code = support_code.replace("ms", "")

    support_map = {
        "5": 25,
        "10": 50,
        "11": 50,
        "20": 100,
        "21": 100,
        "50": 250,
        "51": 250,
        "100": 500,
        "101": 500,
        "200": 1000,
        "201": 1000,
    }

    spatial_support = support_map.get(support_code)

    if spatial_support is None:
        raise ValueError(f"Unrecognized spatial support code: {support_code} from input: {x}")

    return pd.Series({
        "family": family,
        "spatial support": spatial_support
    })



# def label_input_families(df, input_col="input"):
    
#     df = df.copy()

#     x = df[input_col].astype(str)

#     raw_order = {"DEM": 0, "RGB": 1, "NIR": 2, "NHD": 3,"OSM": 4,}
#     terrain_prefix_order = {"ep": 0, "plc": 1, "prc": 2, "s": 3, "sds": 4,}
#     family_order = {"raw sensor": 0, "terrain feature": 1, "multi-scale": 2, "multimodal": 3}

#     families = []
#     sort_keys = []

#     for val in x:
#         val_str = str(val)
#         val_lower = val_str.lower()
#         val_upper = val_str.upper()


#         # 1. raw sensors
#         if val_upper in raw_order:
#             family = "raw sensor"
#             within_family_key = (raw_order[val_upper], val_upper)


#         # 2. Terrain features
#         elif re.fullmatch(r"(ep|sds)-(5|11|21|51|101|201)", val_lower):
#             prefix, num = val_lower.split("-")
#             family = "terrain feature"
#             within_family_key = (terrain_prefix_order[prefix], int(num))

#         elif re.fullmatch(r"(plc|prc|s)-(5|10|20|50|100|200)", val_lower):
#             prefix, num = val_lower.split("-")
#             family = "terrain feature"
#             within_family_key = (terrain_prefix_order[prefix], int(num))


#         # 3. Multi-scale, but only if no "+"
#         elif "+" not in val_lower and val_lower in {"ep-ms", "plc-ms", "prc-ms", "s-ms", "sds-ms"}:
#             family = "multi-scale"
#             within_family_key = (val_lower,)


#         # 4. Multimodal
#         elif "+" in val_lower:
#             family = "multimodal"
#             within_family_key = (val_lower,)

#         families.append(family)
#         sort_keys.append((family_order[family], within_family_key))

#     df["family"] = families
#     df["input_sort_key"] = sort_keys

#     return df




def parse_features(x):
    """Function to parse modality/input strings into modality family and ground sampling distance (GSD) or multi-scale (MS) configurations."""
    x_split = x.split('-')
    if len(x_split) == 1:
        return x_split[0], None
    elif len(x_split) == 2:
        return x_split[0], x_split[1]
    else:
        return 'multimodal', 'ms'
    




def spatial_support(x, bins, raw_gsd=5):
    """Function to group GSD features into families of effective area covered."""
    if x is not None:
        if str(x).isnumeric():
            for bin in bins:
                if np.isclose(int(x) * raw_gsd, bin, atol=10):
                    return bin
        else:
            return x
    else:
        return np.nan