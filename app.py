from src.Sentiment_Driven_stock_price_movement_predictor.logger import logging
from src.Sentiment_Driven_stock_price_movement_predictor.exception import CustomException
from src.Sentiment_Driven_stock_price_movement_predictor.components.data_ingestion import DataIngestionConfig,DataIngestion
from src.Sentiment_Driven_stock_price_movement_predictor.components.model_trainer import ModelTrainerConfig,ModelTrainer
import sys
if __name__=="__main__":
    logging.info("execution has started")
    
    try:
        data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_config()
        #model_trainer_config=ModelTrainerConfig()
        #model_trainer=ModelTrainer()
    except Exception as e:
        raise CustomException(str(e),sys)