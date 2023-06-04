# Nikita Oltyan
# CV 22-1m

import requests
import click
import zipfile
import os


def download_default_zip():
    folder_path = "../data/raw"
    zip_url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"

    # Check if default zip file already downloaded
    if os.path.isfile(f'{folder_path}/hymenoptera_data.zip'):
        return

    os.makedirs(folder_path, exist_ok=True)
    response = requests.get(zip_url)
    filename = zip_url.split('/')[-1]
    file_path = os.path.join(folder_path, filename)

    # Write the downloaded content to the file
    with open(file_path, 'wb') as file:
        file.write(response.content)


def unzip_default_zip():
    zip_path = "../data/raw/hymenoptera_data.zip"
    extract_path = "../data/raw"

    assert os.path.isfile(zip_path), "File isn't exist"
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

download_default_zip()
unzip_default_zip()