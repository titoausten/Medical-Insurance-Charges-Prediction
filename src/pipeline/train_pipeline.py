import sys
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __int__(self):
        pass

    def run_pipeline(self):
        try:
            ingest = DataIngestion()
            train_data, test_data = ingest.initiate_data_ingestion()

            transform = DataTransformation()
            train_arr, test_arr, _ = transform.initiate_data_transformation(train_data, test_data)

            trainer = ModelTrainer()
            print(trainer.initiate_model_trainer(train_arr, test_arr))

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run_pipeline()