import paths
import os
import requests
from zipfile import ZipFile
import zenodo_get
from tqdm import tqdm

def download_from_harvard_dataverse(zip_path, API_KEY, DOI):
    SERVER_URL = "https://dataverse.harvard.edu"

    if os.path.exists(zip_path):
        os.remove(zip_path)

    url = SERVER_URL + '/api/access/dataset/:persistentId/?persistentId=' + DOI
    headers = {
        'X-Dataverse-key': API_KEY,
    }

    download_with_progress_bar(output_filename=zip_path, url=url, header=headers)
    extract_zip_file(zip_path)


def extract_zip_file(zip_path: str):
    extension_ind = zip_path.find(".zip")
    folder_path = zip_path[:extension_ind] + "/"

    with ZipFile(zip_path, 'r') as zObject:
        zObject.extractall(path=folder_path)

    os.remove(zip_path)

def download_from_zenodo(zip_path, doi, file=None):
    base_url = 'https://zenodo.org/record/' + str(doi)
    if file is not None:
        base_url += '/files/' + file

    download_with_progress_bar(output_filename=zip_path, url=base_url)

    extract_zip_file(zip_path)

def download_with_progress_bar(output_filename, url, header=None):
    response = requests.get(url, headers=header, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(output_filename, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")

if __name__=="__main__":
    # download_from_zenodo(zip_path=paths.HOME_PATH + "../Thingiverse_STL_Dataset.zip", doi=1098527, file="all-stl.zip")

    API_KEY = "557f7295-4ee9-4221-b5bb-85b4e86863dc"
    DOI_STL = "doi:10.7910/DVN/S20TXZ"
    download_from_harvard_dataverse(zip_path=paths.HOME_PATH + "../Onshape_STL_Dataset.zip", API_KEY=API_KEY, DOI=DOI_STL)

# zenodo_url = 'https://zenodo.org/record/1098527/files/all-stl.zip'
# download_with_progress_bar(output_filename=paths.HOME_PATH + "../zenodo_zip.zip", url=zenodo_url)


# def save_and_extract_zip_response(response, zip_path):
#     with open(zip_path, "wb") as file:
#         file.write(response.content)
#
#     extract_zip_file(zip_path)

# API_KEY = "557f7295-4ee9-4221-b5bb-85b4e86863dc"
# SERVER_URL = "dataverse.harvard.edu"
# DOI_STL = "doi:10.7910/DVN/FCHHAY"
#
# zip_path = paths.HOME_PATH + "test_zip.zip"
# if os.path.exists(zip_path):
#     os.remove(zip_path)
#
# url = SERVER_URL + '/api/access/dataset/:persistentId/?persistentId=' + DOI_STL
# headers = {
#     'X-Dataverse-key': API_KEY,
# }
#
# response = requests.get(
#     'http://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/FCHHAY',
#     headers=headers,
# )