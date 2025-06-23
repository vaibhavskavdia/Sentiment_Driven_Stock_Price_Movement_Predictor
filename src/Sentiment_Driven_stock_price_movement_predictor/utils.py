import os 
import sys
import yfinance as yf
import requests
import pandas as pd
import time 
import pickle
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.Sentiment_Driven_stock_price_movement_predictor.exception import CustomException
from src.Sentiment_Driven_stock_price_movement_predictor.logger import logging

#API'S and other variables
Alpha_vantage_api= "LFGETE8EEE8RNKB4"
NewsAPI_Key="623831c761644c8a9ce7198b0d57cf3f"
tickers = ["AAPL", "TSLA", "MSFT", "NVDA", "JPM"]
def download_intraday_data(tickers, period="5d", interval="5m", save_path="data/stocks"):
    try:
        """
        This downloads intraday data for a list of tickers and saves them as CSVs.

        Args:
            tickers (list): List of stock tickers.
            period (str): Time period (e.g., "5d", "1mo").
            interval (str): Interval (e.g., "1m", "5m", "15m").
            save_path (str): Folder to save CSVs.
        """
        os.makedirs(save_path, exist_ok=True)

        for ticker in tickers:
            print(f"Downloading {ticker}...")
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if not df.empty:
                df.reset_index(inplace=True)
                df["Ticker"] = ticker
                filename = os.path.join(save_path, f"{ticker}_intraday.csv")
                df.to_csv(filename, index=False)
                print(f"Saved to {filename}")
            else:
                print(f"⚠️ No data returned for {ticker}.")
    except Exception as e:
        raise CustomException(str(e),sys)


def fetch_newsapi_headlines(query, api_key, from_date=None, to_date=None, page_size=100):
    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "apiKey": NewsAPI_Key
        }

    response = requests.get(url, params=params)
    data = response.json()

    if data["status"] != "ok":
        raise Exception(f"Error: {data.get('message', 'Unknown error')}")

    articles = data["articles"]
    return pd.DataFrame([{
            "ticker": query,
            "title": article["title"],
            "publishedAt": article["publishedAt"],
            "source": article["source"]["name"]} for article in articles])
   
def Creating_csv_headline():
    
    all_news = pd.DataFrame()

    for ticker in tickers:
        df = fetch_newsapi_headlines(
            query=ticker,
            api_key=NewsAPI_Key,
            from_date="2025-06-13",
            to_date="2025-06-20"
            )
        all_news = pd.concat([all_news, df], ignore_index=True)

        all_news.to_csv("data/stocks/newsapi_headlines.csv", index=False)


#if you want to use alpha vantage instead of yfinance
def download_intraday_data_alpha_vantage(ticker, interval="5min", outputsize="full", sleep_time=15):
    ts = TimeSeries(key=Alpha_vantage_api, output_format='pandas')
    
    data, meta_data = ts.get_intraday(symbol=ticker, interval=interval, outputsize=outputsize)

    
    data.reset_index(inplace=True)
    data.rename(columns={
        'date': 'timestamp',
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    }, inplace=True)

    data["ticker"] = ticker
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data.sort_values("timestamp", inplace=True)
    
    time.sleep(sleep_time)
    
    return data

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
    
        with open (file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(str(e),sys)

def metrics_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(str(e),sys)