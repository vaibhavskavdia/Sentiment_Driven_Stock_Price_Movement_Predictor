import os
import sys

from dataclasses import dataclass
from src.Sentiment_Driven_stock_price_movement_predictor.exception import CustomException
from src.Sentiment_Driven_stock_price_movement_predictor.logger import logging
from src.Sentiment_Driven_stock_price_movement_predictor.utils import download_intraday_data
from sklearn.model_selection import train_test_split
import yfinance as yf

tickers = ["AAPL", "TSLA", "MSFT", "NVDA", "JPM"]
download_intraday_data(tickers)
@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join("artifacts","train.csv")
    test_data_path=os.path.join("artifacts","test.csv")
    raw_data_path=os.path.join("artifacts","raw.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_config(self):
        try:
            df=download_intraday_data(tickers)
            logging.info("reading from yf database")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("data ingestion is completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(str(e),sys)
        