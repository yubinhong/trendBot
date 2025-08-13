import requests
import json
import os
import mysql.connector
import numpy as np
from datetime import datetime
import time
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force stdout to be unbuffered
sys.stdout.reconfigure(line_buffering=True)

# Load environment variables for security
TAAPI_KEY = os.getenv('TAAPI_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_DB = os.getenv('MYSQL_DB')

if not all([TAAPI_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB]):
    logger.error("Missing required environment variables.")
    raise ValueError("Missing required environment variables.")

logger.info("Starting crypto trend bot...")
logger.info(f"Monitoring symbols: BTC/USDT, ETH/USDT")
logger.info(f"MySQL Host: {MYSQL_HOST}")
logger.info(f"Telegram Chat ID: {TELEGRAM_CHAT_ID}")

def fetch_taapi_data(symbol):
    url = "https://api.taapi.io/bulk"
    payload = {
        "secret": TAAPI_KEY,
        "construct": {
            "exchange": "binance",
            "symbol": symbol,
            "interval": "1d",
            "indicators": [
                {"indicator": "sma", "period": 50, "id": "sma50"},
                {"indicator": "sma", "period": 200, "id": "sma200"},
                {"indicator": "dmi", "id": "dmi"},  # Default period 14 for ADX, +DI, -DI
                {"indicator": "bbands", "results": 252, "id": "bbands"},  # Default period 20, historical for percentile
                {"indicator": "atr", "results": 252, "id": "atr"}  # Default period 14, historical for percentile
            ]
        }
    }
    logger.debug(f"Fetching data for {symbol}")
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        logger.error(f"TAAPI error for {symbol}: {response.text}")
        raise Exception(f"TAAPI error for {symbol}: {response.text}")
    logger.debug(f"Successfully fetched data for {symbol}")
    return response.json()['data']

def parse_indicators(data):
    indicators = {}
    for item in data:
        ind_id = item['id']
        result = item['result']
        if isinstance(result, list):
            # For historical data (bbands, atr), assume ordered oldest to newest, last is current
            if ind_id == 'bbands':
                bandwidths = [
                    (r['valueUpperBand'] - r['valueLowerBand']) / r['valueMiddleBand'] * 100
                    if r['valueMiddleBand'] != 0 else 0 for r in result
                ]
                indicators['bandwidths'] = bandwidths
            elif ind_id == 'atr':
                atr_values = [r['value'] for r in result]
                indicators['atr_values'] = atr_values
        else:
            # For single results
            if ind_id in ['sma50', 'sma200']:
                indicators[ind_id] = result['value']
            elif ind_id == 'dmi':
                indicators['adx'] = result['adx']
                indicators['pdi'] = result['pdi']
                indicators['mdi'] = result['mdi']
    return indicators

def determine_trend(indicators):
    adx = indicators['adx']
    pdi = indicators['pdi']
    mdi = indicators['mdi']
    sma50 = indicators['sma50']
    sma200 = indicators['sma200']
    
    # Bandwidth analysis
    bandwidths = indicators.get('bandwidths', [])
    if len(bandwidths) >= 20:
        avg_last_20_bw = np.mean(bandwidths[-20:])
        percentile_30_bw = np.percentile(bandwidths, 30)
        low_bw = avg_last_20_bw < percentile_30_bw
    else:
        low_bw = False  # Insufficient data
    
    # ATR analysis
    atr_values = indicators.get('atr_values', [])
    if atr_values:
        current_atr = atr_values[-1]
        percentile_30_atr = np.percentile(atr_values, 30)
        low_atr = current_atr < percentile_30_atr
    else:
        low_atr = False
    
    if adx > 25 and pdi > mdi and sma50 > sma200:
        return "上涨趋势"
    elif adx > 25 and mdi > pdi and sma50 < sma200:
        return "下跌趋势"
    elif adx <= 25 and (low_bw or low_atr):
        return "区间/波动小"
    else:
        return "未知"

def generate_insight(symbol, trend):
    if symbol == "BTC/USDT":
        if trend == "上涨趋势":
            return "基于当前强势ADX和+DI主导，短期内预计继续上涨，可能测试更高阻力位。"
        elif trend == "下跌趋势":
            return "当前-DI主导且SMA交叉向下，短期可能延续下跌，关注支撑位。市场仍有潜在反弹机会。"
        elif trend == "区间/波动小":
            return "ADX低位且波动率低，预计短期内维持震荡，等待突破信号。"
        else:
            return "趋势不明，建议观察更多数据。"
    elif symbol == "ETH/USDT":
        if trend == "上涨趋势":
            return "基于当前强势ADX和+DI主导，短期内预计继续上涨，可能测试更高阻力位。"
        elif trend == "下跌趋势":
            return "当前-DI主导且SMA交叉向下，短期可能延续下跌，关注支撑位。市场仍有潜在反弹机会。"
        elif trend == "区间/波动小":
            return "ADX低位且波动率低，预计短期内维持震荡，等待突破信号。"
        else:
            return "趋势不明，建议观察更多数据。"

def store_to_mysql(symbol, indicators, trend):
    try:
        logger.debug(f"Connecting to MySQL for {symbol}")
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crypto_trends (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(20),
                timestamp DATETIME,
                adx FLOAT,
                pdi FLOAT,
                mdi FLOAT,
                sma50 FLOAT,
                sma200 FLOAT,
                bandwidth FLOAT,
                atr FLOAT,
                trend VARCHAR(50)
            )
        """)
        # Insert data
        now = datetime.now()
        current_bw = indicators['bandwidths'][-1] if 'bandwidths' in indicators else None
        current_atr = indicators['atr_values'][-1] if 'atr_values' in indicators else None
        cursor.execute("""
            INSERT INTO crypto_trends (symbol, timestamp, adx, pdi, mdi, sma50, sma200, bandwidth, atr, trend)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (symbol, now, indicators['adx'], indicators['pdi'], indicators['mdi'], indicators['sma50'], indicators['sma200'], current_bw, current_atr, trend))
        conn.commit()
        cursor.close()
        conn.close()
        logger.debug(f"Successfully stored data for {symbol}")
    except Exception as e:
        logger.error(f"MySQL error for {symbol}: {str(e)}")
        raise

def send_to_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={message}"
        response = requests.get(url)
        if response.status_code != 200:
            logger.error(f"Telegram send error: {response.text}")
        else:
            logger.info("Telegram message sent successfully")
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {str(e)}")

if __name__ == "__main__":
    symbols = ["BTC/USDT", "ETH/USDT"]
    last_trends = {sym: None for sym in symbols}
    
    logger.info("Bot started successfully, beginning monitoring loop...")
    
    while True:
        try:
            logger.info("Starting new analysis cycle...")
            trends = {}
            insights = {}
            trend_changed = False
            
            for symbol in symbols:
                logger.info(f"Analyzing {symbol}...")
                ta_data = fetch_taapi_data(symbol)
                indicators = parse_indicators(ta_data)
                trend = determine_trend(indicators)
                trends[symbol] = trend
                insights[symbol] = generate_insight(symbol, trend)
                store_to_mysql(symbol, indicators, trend)
                
                if trend != last_trends[symbol]:
                    trend_changed = True
                    last_trends[symbol] = trend
                    logger.info(f"{symbol} 趋势变化: {last_trends.get(symbol, 'None')} -> {trend}")
                else:
                    logger.info(f"{symbol} 趋势保持: {trend}")
            
            if trend_changed:
                message = ""
                for symbol in symbols:
                    message += f"当前{symbol.split('/')[0]}趋势: {trends[symbol]}\n理解与预测: {insights[symbol]}\n\n"
                logger.info("Trend changed, sending Telegram notification...")
                send_to_telegram(message.strip())
                logger.info(f"Notification sent: {message}")
            else:
                logger.info("No trend changes detected, skipping notification")
                
            logger.info("Analysis cycle completed, waiting 60 seconds...")
            
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            logger.info("Continuing after error...")
            
        time.sleep(60)  # 每分钟运行一次