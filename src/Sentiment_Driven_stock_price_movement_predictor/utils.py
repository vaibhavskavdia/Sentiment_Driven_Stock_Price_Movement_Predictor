import os 
import sys


from src.Sentiment_Driven_stock_price_movement_predictor.exception import CustomException
from src.Sentiment_Driven_stock_price_movement_predictor.logger import logging

import yfinance as yf





def download_intraday_data(tickers, period="5d", interval="5m", save_path="data/stocks"):
    try:
        """
        Downloads intraday data for a list of tickers and saves them as CSVs.

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

# Example usage:


    
    