import os
import sys
import requests
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from src.Sentiment_Driven_stock_price_movement_predictor.exception import CustomException
from src.Sentiment_Driven_stock_price_movement_predictor.logger import logging
from src.Sentiment_Driven_stock_price_movement_predictor.utils import download_intraday_data
#from src.Sentiment_Driven_stock_price_movement_predictor.utils import download_intraday_data_alpha_vantage
from sklearn.model_selection import train_test_split
import yfinance as yf
from src.Sentiment_Driven_stock_price_movement_predictor.utils import Creating_csv_headline
all_dfs=[]
tickers = ["AAPL", "TSLA", "MSFT", "NVDA", "JPM"]
download_intraday_data(tickers)
@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join("data/stocks","train.csv")
    test_data_path=os.path.join("data/stocks","test.csv")
    raw_data_path=os.path.join("data/stocks","raw.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_config(self):
        try:
            df1=download_intraday_data(tickers)
            df2=Creating_csv_headline()
            logging.info("reading from yf database")
            #for ticker in tickers:
                #print(f"Downloading: {ticker}")
                #df = download_intraday_data_alpha_vantage(ticker)
                #df.to_csv(f"data/stocks/{ticker}_intraday_alpha.csv", index=False)
                #all_dfs.append(df)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df1.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            df2.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_set,test_set=train_test_split(df1,test_size=0.2,random_state=42)
            train_set,test_set=train_test_split(df2,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("data ingestion is completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(str(e),sys)
        

    