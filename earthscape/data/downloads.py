
import os
import requests
import zipfile


def download_zip(url, output_dir):
    """
    Function to download zip file, extract contents in the specified directory, and delete the zip file.

    Parameters
    ----------
    url : str
        Download URL for zip file.
    output_dir : str
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
            print('Reponse code not 200 for downloading .zip...')
    except:
        print('Error downloading .zip...')



def download_tif(url, output_path):
    """
    Function to download TIFF file from a specified URL.

    Parameters
    ----------
    url : str
        Download URL for GeoTIFF file.
    output_path : str
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
            print('Reponse code for URL not 200...')
    except:
        print(f"Error connecting to URL...\n{url}")

