
import os
# import glob
import requests
import zipfile
import pandas as pd
import geopandas as gpd
# import shapely
from pathlib import Path
from urllib.parse import urlparse



def download_zip(url, output_dir):
    """
    Function to download zip file from a specified URL, extract 
    contents in the specified output directory, and delete the 
    zip file.

    Parameters
    ----------
    url : str
        Download URL for zip file.
    output_dir : str or pathlib.Path
        Directory path to save zip file and extract contents.

    Returns
    -------
    None
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        zip_path = os.path.join(output_dir, 'download.zip')
        if response.status_code == 200:
            with open(zip_path, 'wb') as zip:
                zip.write(response.content)
            with zipfile.ZipFile(zip_path, 'r') as zip:
                zip.extractall(output_dir)
            os.remove(zip_path)
        else:
            print('Reponse code not 200...')
    except:
        print('Error connecting to download URL...')



def download_tif(url, output_path):
    """
    Function to download TIFF file from a specified URL.

    Parameters
    ----------
    url : str
        Download URL for GeoTIFF file.
    output_path : str or pathlib.Path
        Path to save GeoTIFF.

    Returns
    -------
    None
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        if response.status_code == 200:
            with open(output_path, 'wb') as tif:
                tif.write(response.content)
        else:
            print('Reponse code not 200...')
    except:
        print(f"Error connecting to URL...")






def ky_index_tiles(aoi_path, 
                   url_field="Phase1_AWS_url", 
                   layer_url=r"https://kygisserver.ky.gov/arcgis/rest/services/WGS84WM_Services/KY_Data_Tiles_DEM_WGS84WM/MapServer/0", 
                   tile_name_field="Tile_Name", 
                   page_size=1000, 
                   pad=0):
    """
    Retrieve Kentucky DEM tiles intersecting an AOI. The AOI is 
    reprojected to EPSG:3857, its bounding box (optionally
    padded) is used to query the ArcGIS REST tile index layer, and
    results are paginated until all matching tiles are returned.

    Parameters
    ----------
    aoi_path : str or pathlib.Path
        Path to an AOI vector dataset readable by GeoPandas.
    url_field : str
        Field name containing the tile download URL.
    layer_url : str
        ArcGIS REST layer URL (ending in /MapServer/<layer_id>).
    tile_name_field : str
        Field name containing the tile name.
    page_size : int
        Number of records requested per page from the REST service.
    pad : float
        Padding (in EPSG:3857 units) applied to the AOI bounding box.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame in EPSG:3857 with columns ['tile', 'url', 'geometry'].
    """


    # setup query for ArcGIS web service
    query_url = f"{layer_url}/query"

    # read AOI and move to Web Mercator CRS (to match web service)...
    aoi = gpd.read_file(aoi_path)              # read AOI GeoJSON
    aoi_3857 = aoi.to_crs(3857)                # reproject to Web Mercator EPSG:3857
    aoi_geom = aoi_3857.geometry.union_all()   # confirms AOI is one polygon
    minx, miny, maxx, maxy = aoi_geom.bounds   # get bounding box coordinates
    
    # add padding to force inclusion of edge tiles
    minx -= pad; miny -= pad; maxx += pad; maxy += pad

    # page through features in the AOI bbox...
    chunks = []
    offset = 0
    while True:
        params = {
            "f": "geojson",
            "where": "1=1",
            "outFields": f"{tile_name_field}, {url_field}",
            "returnGeometry": "true",
            "outSR": "3857",
            "inSR": "3857",
            "geometryType": "esriGeometryEnvelope",
            "geometry": f"{minx},{miny},{maxx},{maxy}",
            "spatialRel": "esriSpatialRelIntersects",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "orderByFields": "OBJECTID",
        }

        # send query request...
        r = requests.post(query_url, data=params, timeout=120)
        r.raise_for_status()

        # read & append response; discard if empty...
        gdf = gpd.read_file(r.text)
        if gdf.empty:
            break
        chunks.append(gdf)

        # determine if additional queries needed...
        if len(gdf) < page_size:
            break
        offset += page_size

    # return blank gdf...
    if not chunks:
        return gpd.GeoDataFrame(columns=[tile_name_field, url_field, "geometry"], geometry="geometry", crs="EPSG:3857")

    # return gdf of tile names...
    tiles = gpd.GeoDataFrame(pd.concat(chunks, ignore_index=True))
    out = tiles[[tile_name_field, url_field, "geometry"]].copy()
    out = out.dropna(subset=[url_field])
    out.rename(columns={tile_name_field: 'tile', url_field:'url'}, inplace=True)

    return out



def get_kyfromabove_tiles(index_path, id_field, url_field, output_dir):
    """
    Download KYFromAbove tiles listed in an index dataset. Reads a GeoJSON 
    (or other GeoPandas-readable vector file) containing
    tile identifiers and download URLs, then iterates through each record
    and downloads the referenced resource. Files with a ``.tif`` extension
    are downloaded directly to ``output_dir`` using the tile id as the
    output filename. Files with a ``.zip`` extension are downloaded and
    handled via ``download_zip``. Other extensions are reported but not
    downloaded.

    Parameters
    ----------
    index_path : str or path-like
        Path to a vector dataset readable by GeoPandas containing tile records.
    id_field : str
        Field name containing the tile identifier used for output naming.
    url_field : str
        Field name containing the download URL.
    output_dir : str or path-like
        Directory where downloaded files will be written.

    Returns
    -------
    None
        Files are written to ``output_dir``.
    """

    # read KyFromAbove data tile index GeoJSON as gdf
    gdf = gpd.read_file(index_path)
    
    # iterate through tiles...
    for _, tile in gdf.iterrows():
        tile_id = tile[id_field]
        url = tile[url_field]
        ext = Path(urlparse(url).path).suffix.lower()

        # download TIFF images...
        if ext == 'tif':
            output_path = f"{output_dir}/{tile_id}.tif"
            download_tif(url, output_path)

        # download .ZIP files...
        elif ext == 'zip':
            download_zip(url, output_dir)

        # print failure mode...
        else:
            print(f"Unsupported download type for tile {tile_id}: {ext or '(no extension)'}")


        

        
