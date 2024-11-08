import os
import urllib.request as request
import zipfile
from Steganalysis import logger
from pathlib import Path
import logging
from Steganalysis.utils.common import get_size
from Steganalysis.entity.config_entity import DataIngestionConfig
from pathlib import Path

# from kaggle.api.kaggle_api_extended import KaggleApi  # Import the Kaggle API client
# # Set up custom path for Kaggle API token
# os.environ["KAGGLE_CONFIG_DIR"] = r"C:\Users\haris\Documents\kaggle.json"

# # DOWNLOAD FROM KAGGLE

# logger = logging.getLogger(__name__)

# class DataIngestion:
#     def __init__(self, config):
#         self.config = config
#         self.api = KaggleApi()
#         self.api.authenticate()  # Authenticate using Kaggle API credentials

#     def download_file(self):
#         # Check if file already exists
#         if not os.path.exists(self.config.local_data_file):
#             os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)
            
#             # Download from Kaggle competition
#             logger.info(f"Downloading dataset from Kaggle competition: {self.config.kaggle_competition}")
#             self.api.competition_download_files(
#                 competition=self.config.kaggle_competition,
#                 path=os.path.dirname(self.config.local_data_file),
#                 quiet=False
#             )
#             logger.info(f"Downloaded dataset from {self.config.kaggle_competition} to {self.config.local_data_file}")
#         else:
#             logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  

#     def extract_zip_file(self):
#         """
#         Extracts the zip file into the data directory.
#         """
#         unzip_path = self.config.unzip_dir
#         os.makedirs(unzip_path, exist_ok=True)
        
#         with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
#             zip_ref.extractall(unzip_path)
#             logger.info(f"Extracted zip file to {unzip_path}")

# # DOWNLOAD FROM GITHUB
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  


    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

def get_size(path: Path):
    """Utility function to get file size in MB"""
    size = round(path.stat().st_size / (1024 * 1024), 2)
    return f"{size} MB"
