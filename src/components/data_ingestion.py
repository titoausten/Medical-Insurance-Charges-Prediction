import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import urllib.request as request
from src.utils import get_size


@dataclass
class DataIngestionConfig:
    data_path: str = '../../artifacts'
    train_data_path: str = os.path.join(data_path, "train.csv")
    test_data_path: str = os.path.join(data_path, "test.csv")
    raw_data_path: str = os.path.join(data_path, "data.csv")
    data_download_url: str = "https://raw.githubusercontent.com/titoausten/Global-AI-Hub-Machine-Learning-Bootcamp/main/insurance.csv"


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def download_file(self):
        if not os.path.exists(self.ingestion_config.data_path):

            os.makedirs(self.ingestion_config.data_path, exist_ok=True)
            filename, headers = request.urlretrieve(
                url=self.ingestion_config.data_download_url,
                filename=self.ingestion_config.raw_data_path
            )
            logging.info(f"{filename} download! with following info: \n{headers}")

            return self.ingestion_config.raw_data_path
        else:
            logging.info(f"File already exists of size: {get_size(self.ingestion_config.raw_data_path)}")

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started...")
        try:
            # df = pd.read_csv('../../insurance.csv')

            df = pd.read_csv(self.download_file())

            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion completed...")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e, sys)
