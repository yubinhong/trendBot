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

def fetch_taapi_data(symbol, max_retries=3):
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
                {"indicator": "bbands", "results": 20, "id": "bbands"},  # Limited to 20 results
                {"indicator": "atr", "results": 20, "id": "atr"}  # Limited to 20 results
            ]
        }
    }
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Fetching data for {symbol} (attempt {attempt + 1}/{max_retries})")
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                logger.debug(f"Successfully fetched data for {symbol}")
                return response.json()['data']
            elif response.status_code == 429:  # Rate limit
                wait_time = (attempt + 1) * 10  # Exponential backoff
                logger.warning(f"Rate limit hit for {symbol}, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"TAAPI error for {symbol}: {response.text}")
                if attempt == max_retries - 1:  # Last attempt
                    raise Exception(f"TAAPI error for {symbol}: {response.text}")
                time.sleep(5)  # Wait before retry
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {symbol}: {str(e)}")
            if attempt == max_retries - 1:  # Last attempt
                raise Exception(f"Request error for {symbol}: {str(e)}")
            time.sleep(5)  # Wait before retry
    
    raise Exception(f"Failed to fetch data for {symbol} after {max_retries} attempts")

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
    
    logger.info(f"Indicators - ADX: {adx:.2f}, +DI: {pdi:.2f}, -DI: {mdi:.2f}, SMA50: {sma50:.2f}, SMA200: {sma200:.2f}")
    
    # Bandwidth analysis (using available data, max 20 points)
    bandwidths = indicators.get('bandwidths', [])
    if len(bandwidths) >= 10:  # Need at least 10 points for meaningful analysis
        avg_recent_bw = np.mean(bandwidths[-5:]) if len(bandwidths) >= 5 else np.mean(bandwidths)
        percentile_30_bw = np.percentile(bandwidths, 30)
        low_bw = avg_recent_bw < percentile_30_bw
    else:
        low_bw = False  # Insufficient data
    
    # ATR analysis (using available data, max 20 points)
    atr_values = indicators.get('atr_values', [])
    if len(atr_values) >= 10:  # Need at least 10 points for meaningful analysis
        current_atr = atr_values[-1]
        percentile_30_atr = np.percentile(atr_values, 30)
        low_atr = current_atr < percentile_30_atr
    else:
        low_atr = False
    
    # Log analysis details
    logger.info(f"Trend analysis - ADX>25: {adx > 25}, +DI>-DI: {pdi > mdi}, SMA50>SMA200: {sma50 > sma200}")
    logger.info(f"Volatility analysis - Low BW: {low_bw}, Low ATR: {low_atr}")
    
    # More nuanced trend detection
    if adx > 25 and pdi > mdi and sma50 > sma200:
        logger.info("Trend decision: 上涨趋势 (Strong uptrend)")
        return "上涨趋势"
    elif adx > 25 and mdi > pdi and sma50 < sma200:
        logger.info("Trend decision: 下跌趋势 (Strong downtrend)")
        return "下跌趋势"
    elif adx > 20 and pdi > mdi * 1.1 and sma50 > sma200:  # Moderate uptrend
        logger.info("Trend decision: 上涨趋势 (Moderate uptrend)")
        return "上涨趋势"
    elif adx > 20 and mdi > pdi * 1.1 and sma50 < sma200:  # Moderate downtrend
        logger.info("Trend decision: 下跌趋势 (Moderate downtrend)")
        return "下跌趋势"
    elif adx <= 25 and (low_bw or low_atr):
        logger.info("Trend decision: 区间/波动小 (Range-bound/Low volatility)")
        return "区间/波动小"
    elif sma50 > sma200 * 1.02:  # SMA50 significantly above SMA200
        logger.info("Trend decision: 上涨趋势 (SMA-based uptrend)")
        return "上涨趋势"
    elif sma50 < sma200 * 0.98:  # SMA50 significantly below SMA200
        logger.info("Trend decision: 下跌趋势 (SMA-based downtrend)")
        return "下跌趋势"
    else:
        logger.info("Trend decision: 未知 (Unknown - mixed signals)")
        return "未知"

def generate_insight(symbol, trend, indicators=None):
    base_insights = {
        "上涨趋势": "基于当前强势ADX和+DI主导，短期内预计继续上涨，可能测试更高阻力位。",
        "下跌趋势": "当前-DI主导且SMA交叉向下，短期可能延续下跌，关注支撑位。市场仍有潜在反弹机会。",
        "区间/波动小": "ADX低位且波动率低，预计短期内维持震荡，等待突破信号。"
    }
    
    if trend in base_insights:
        return base_insights[trend]
    else:
        # Enhanced insight for "未知" trend
        if indicators:
            adx = indicators.get('adx', 0)
            pdi = indicators.get('pdi', 0)
            mdi = indicators.get('mdi', 0)
            sma50 = indicators.get('sma50', 0)
            sma200 = indicators.get('sma200', 0)
            
            details = []
            if adx <= 25:
                details.append(f"ADX较低({adx:.1f})表明趋势强度不足")
            if abs(pdi - mdi) < 5:
                details.append("买卖力量相当，方向不明确")
            if abs(sma50 - sma200) / sma200 < 0.02:  # Less than 2% difference
                details.append("短长期均线接近，缺乏明确方向")
            
            if details:
                return f"当前信号混合：{'; '.join(details)}。建议等待更明确的突破信号。"
        
        return "趋势信号混合，建议观察关键技术位突破情况。"

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
            
            for i, symbol in enumerate(symbols):
                logger.info(f"Analyzing {symbol}...")
                
                # Add delay between requests to avoid rate limiting
                if i > 0:
                    logger.info("Waiting 5 seconds to avoid rate limit...")
                    time.sleep(5)
                
                ta_data = fetch_taapi_data(symbol)
                indicators = parse_indicators(ta_data)
                trend = determine_trend(indicators)
                trends[symbol] = trend
                insights[symbol] = generate_insight(symbol, trend, indicators)
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
                
            logger.info("Analysis cycle completed, waiting 5 minutes...")
            
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            logger.info("Continuing after error...")
            
        time.sleep(300)  # 每5分钟运行一次