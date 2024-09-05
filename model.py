import os
import pandas as pd
from datetime import datetime, timedelta
import random
from config import data_base_path
import random
import requests
import retrying
from prophet import Prophet

forecast_price = {}

binance_data_path = os.path.join(data_base_path, "binance/futures-klines")
MAX_DATA_SIZE = 1000  # Giới hạn số lượng dữ liệu tối đa khi lưu trữ
INITIAL_FETCH_SIZE = 1000  # Số lượng nến lần đầu tải về

@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5)
def fetch_prices(symbol, interval="1m", limit=1000, start_time=None, end_time=None):
    try:
        base_url = "https://fapi.binance.com"
        endpoint = f"/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        url = base_url + endpoint
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f'Failed to fetch prices for {symbol} from Binance API: {str(e)}')
        raise e

def download_data(token):
    symbols = f"{token.upper()}USDT"
    interval = "5m"
    current_datetime = datetime.now()
    download_path = os.path.join(binance_data_path, token.lower())

    file_path = os.path.join(download_path, f"{token.lower()}_5m_data.csv")
    # file_path = os.path.join(data_base_path, f"{token.lower()}_price_data.csv")

    # Kiểm tra xem file có tồn tại hay không
    if os.path.exists(file_path):
        # Tính thời gian bắt đầu cho 100 cây nến 5 phút
        start_time = int((current_datetime - timedelta(minutes=500)).timestamp() * 1000)
        end_time = int(current_datetime.timestamp() * 1000)
        new_data = fetch_prices(symbols, interval, 1000, start_time, end_time)
    else:
        # Nếu file không tồn tại, tải về số lượng INITIAL_FETCH_SIZE nến
        start_time = int((current_datetime - timedelta(minutes=INITIAL_FETCH_SIZE*5)).timestamp() * 1000)
        end_time = int(current_datetime.timestamp() * 1000)
        new_data = fetch_prices(symbols, interval, INITIAL_FETCH_SIZE, start_time, end_time)

    # Chuyển dữ liệu thành DataFrame
    new_df = pd.DataFrame(new_data, columns=[
        "start_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", 
        "taker_buy_quote_asset_volume", "ignore"
    ])

    # Kiểm tra và đọc dữ liệu cũ nếu tồn tại
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        # Kết hợp dữ liệu cũ và mới
        combined_df = pd.concat([old_df, new_df])
        # Loại bỏ các bản ghi trùng lặp dựa trên 'start_time'
        combined_df = combined_df.drop_duplicates(subset=['start_time'], keep='last')
    else:
        combined_df = new_df

    # Giới hạn số lượng dữ liệu tối đa
    if len(combined_df) > MAX_DATA_SIZE:
        combined_df = combined_df.iloc[-MAX_DATA_SIZE:]

    # Lưu dữ liệu đã kết hợp vào file CSV
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    combined_df.to_csv(file_path, index=False)
    forecast_price[token] = predict_price(token)
    print(f"Updated data for {token} saved to {file_path}. Total rows: {len(combined_df)}")


def predict_price(token, period="20T"):
    download_path = os.path.join(binance_data_path, token.lower())
    file_path = os.path.join(download_path, f"{token.lower()}_5m_data.csv")
    df = pd.read_csv(file_path)
    df['ds'] = pd.to_datetime(df['start_time'], unit='ms')
    df = df[['ds', 'close', 'volume', 'number_of_trades']].rename(columns={'close': 'y'})

    model = Prophet()

    model.add_regressor('volume')
    model.add_regressor('number_of_trades')

    model.fit(df)

    future = model.make_future_dataframe(periods=1, freq=period)

    last_known_volume = df['volume'].iloc[-1]
    last_known_trades = df['number_of_trades'].iloc[-1]

    future['volume'] = last_known_volume
    future['number_of_trades'] = last_known_trades

    forecast = model.predict(future)

    predicted_price = forecast['yhat'].iloc[-1]
    random_percentage = random.uniform(-0.001, 0.001)
    adjusted_price = predicted_price * (1 + random_percentage)
    return adjusted_price

def update_data():
    tokens = ["ETH", "BNB", "ARB"]
    for token in tokens:
        download_data(token)

if __name__ == "__main__":
    update_data()