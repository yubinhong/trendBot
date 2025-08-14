import requests
import json
import os
import mysql.connector
import numpy as np
import talib
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

def fetch_taapi_data(symbol, interval="1h", max_retries=3):
    url = "https://api.taapi.io/bulk"
    payload = {
        "secret": TAAPI_KEY,
        "construct": {
            "exchange": "binance",
            "symbol": symbol,
            "interval": interval,
            "indicators": [
                {"indicator": "sma", "period": 50, "id": "sma50"},
                {"indicator": "sma", "period": 200, "id": "sma200"},
                {"indicator": "dmi", "id": "dmi"},  # Default period 14 for ADX, +DI, -DI
                {"indicator": "bbands", "results": 20, "id": "bbands"},  # Limited to 20 results
                {"indicator": "atr", "results": 20, "id": "atr"},  # Limited to 20 results
                {"indicator": "price", "id": "price"}  # Get current price data
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
            elif ind_id == 'price':
                indicators['price'] = result['value']
                indicators['open'] = result.get('open', result['value'])
                indicators['high'] = result.get('high', result['value'])
                indicators['low'] = result.get('low', result['value'])
                indicators['close'] = result['value']
                indicators['volume'] = result.get('volume', 0)
    return indicators

def get_historical_data(symbol, minutes_back):
    """从数据库获取历史数据用于多时间框架分析"""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, price, adx, pdi, mdi, sma50, sma200, bandwidth, atr
            FROM crypto_5min_data 
            WHERE symbol = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL %s MINUTE)
            ORDER BY timestamp DESC
            LIMIT %s
        """, (symbol, minutes_back, minutes_back // 5))  # 5分钟数据，所以除以5
        
        data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return data
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
        return []

def determine_trend(indicators, timeframe="5m"):
    """基于指标和时间框架判断趋势"""
    adx = indicators['adx']
    pdi = indicators['pdi']
    mdi = indicators['mdi']
    sma50 = indicators['sma50']
    sma200 = indicators['sma200']
    
    logger.info(f"[{timeframe}] Indicators - ADX: {adx:.2f}, +DI: {pdi:.2f}, -DI: {mdi:.2f}, SMA50: {sma50:.2f}, SMA200: {sma200:.2f}")
    
    # 根据时间框架调整阈值
    adx_strong_threshold = 25
    adx_moderate_threshold = 20
    sma_diff_threshold = 0.02
    
    if timeframe == "15m":
        adx_strong_threshold = 35  # 15分钟需要更强的信号
        adx_moderate_threshold = 30
        sma_diff_threshold = 0.008
    elif timeframe == "1h":
        adx_strong_threshold = 30  # 1小时需要更强的信号
        adx_moderate_threshold = 25
        sma_diff_threshold = 0.01
    elif timeframe == "4h":
        adx_strong_threshold = 25
        adx_moderate_threshold = 20
        sma_diff_threshold = 0.015
    elif timeframe == "1d":
        adx_strong_threshold = 20  # 长期时间框架可以用较低阈值
        adx_moderate_threshold = 15
        sma_diff_threshold = 0.02
    elif timeframe == "1w":
        adx_strong_threshold = 18
        adx_moderate_threshold = 12
        sma_diff_threshold = 0.03
    
    # Bandwidth analysis
    bandwidths = indicators.get('bandwidths', [])
    if len(bandwidths) >= 10:
        avg_recent_bw = np.mean(bandwidths[-5:]) if len(bandwidths) >= 5 else np.mean(bandwidths)
        percentile_30_bw = np.percentile(bandwidths, 30)
        low_bw = avg_recent_bw < percentile_30_bw
    else:
        low_bw = False
    
    # ATR analysis
    atr_values = indicators.get('atr_values', [])
    if len(atr_values) >= 10:
        current_atr = atr_values[-1]
        percentile_30_atr = np.percentile(atr_values, 30)
        low_atr = current_atr < percentile_30_atr
    else:
        low_atr = False
    
    logger.info(f"[{timeframe}] Trend analysis - ADX>{adx_strong_threshold}: {adx > adx_strong_threshold}, +DI>-DI: {pdi > mdi}, SMA50>SMA200: {sma50 > sma200}")
    
    # 趋势判断逻辑
    if adx > adx_strong_threshold and pdi > mdi and sma50 > sma200:
        logger.info(f"[{timeframe}] Trend decision: 上涨趋势 (Strong uptrend)")
        return "上涨趋势"
    elif adx > adx_strong_threshold and mdi > pdi and sma50 < sma200:
        logger.info(f"[{timeframe}] Trend decision: 下跌趋势 (Strong downtrend)")
        return "下跌趋势"
    elif adx > adx_moderate_threshold and pdi > mdi * 1.1 and sma50 > sma200:
        logger.info(f"[{timeframe}] Trend decision: 上涨趋势 (Moderate uptrend)")
        return "上涨趋势"
    elif adx > adx_moderate_threshold and mdi > pdi * 1.1 and sma50 < sma200:
        logger.info(f"[{timeframe}] Trend decision: 下跌趋势 (Moderate downtrend)")
        return "下跌趋势"
    elif adx <= adx_strong_threshold and (low_bw or low_atr):
        logger.info(f"[{timeframe}] Trend decision: 区间/波动小 (Range-bound)")
        return "区间/波动小"
    elif sma50 > sma200 * (1 + sma_diff_threshold):
        logger.info(f"[{timeframe}] Trend decision: 上涨趋势 (SMA-based uptrend)")
        return "上涨趋势"
    elif sma50 < sma200 * (1 - sma_diff_threshold):
        logger.info(f"[{timeframe}] Trend decision: 下跌趋势 (SMA-based downtrend)")
        return "下跌趋势"
    else:
        logger.info(f"[{timeframe}] Trend decision: 未知 (Mixed signals)")
        return "未知"

def calculate_indicators_from_5min_data(symbol, timeframe_minutes):
    """基于5分钟数据计算指定时间框架的技术指标"""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        
        # 计算需要的5分钟数据量
        # 为了计算SMA200，我们需要200个目标时间框架的周期
        # 每个目标时间框架需要 timeframe_minutes/5 个5分钟数据
        periods_per_timeframe = timeframe_minutes // 5
        target_periods = 250  # 目标时间框架周期数（比200多一些以确保数据充足）
        periods_needed = periods_per_timeframe * target_periods
        
        logger.debug(f"Fetching {periods_needed} 5min periods for {timeframe_minutes}min analysis")
        
        cursor.execute("""
            SELECT timestamp, close_price, high_price, low_price, open_price, volume
            FROM crypto_5min_data 
            WHERE symbol = %s AND close_price > 0 AND high_price > 0 AND low_price > 0
            ORDER BY timestamp DESC
            LIMIT %s
        """, (symbol, periods_needed))
        
        raw_data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if len(raw_data) < periods_per_timeframe * 50:  # 至少需要50个目标周期的数据
            logger.warning(f"Insufficient 5min data for {symbol} {timeframe_minutes}m analysis: {len(raw_data)} records")
            return None
        
        logger.debug(f"Retrieved {len(raw_data)} 5min records for {symbol}")
        
        # 将5分钟数据聚合为目标时间框架
        aggregated_data = aggregate_5min_to_timeframe(raw_data, timeframe_minutes)
        
        if len(aggregated_data) < 200:  # 需要至少200个周期来计算SMA200
            logger.warning(f"Insufficient aggregated data for {symbol} {timeframe_minutes}m: {len(aggregated_data)} periods")
            return None
        
        logger.debug(f"Aggregated to {len(aggregated_data)} {timeframe_minutes}min periods")
        
        # 计算技术指标
        indicators = calculate_technical_indicators(aggregated_data)
        
        if indicators is None:
            logger.warning(f"Failed to calculate indicators for {symbol} {timeframe_minutes}m")
            return None
        
        logger.debug(f"Successfully calculated indicators for {symbol} {timeframe_minutes}m")
        return indicators
        
    except Exception as e:
        logger.error(f"Error calculating indicators for {symbol} {timeframe_minutes}m: {str(e)}")
        return None

def aggregate_5min_to_timeframe(raw_data, timeframe_minutes):
    """将5分钟数据聚合为指定时间框架的OHLCV数据"""
    if not raw_data:
        return []
    
    # 按时间框架分组数据
    periods_per_timeframe = timeframe_minutes // 5
    aggregated = []
    
    # 反转数据，从最旧到最新
    raw_data = list(reversed(raw_data))
    
    logger.debug(f"Aggregating {len(raw_data)} 5min periods to {timeframe_minutes}min timeframe")
    logger.debug(f"Periods per timeframe: {periods_per_timeframe}")
    
    for i in range(0, len(raw_data), periods_per_timeframe):
        period_data = raw_data[i:i + periods_per_timeframe]
        
        # 对于不完整的周期，如果数据量足够（至少一半），也包含进来
        if len(period_data) < max(1, periods_per_timeframe // 2):
            continue
        
        try:
            # 聚合OHLCV，确保数据类型正确
            timestamp = period_data[0][0]  # 使用周期开始时间
            open_price = float(period_data[0][4])  # 第一个5分钟的开盘价
            close_price = float(period_data[-1][1])  # 最后一个5分钟的收盘价
            
            # 过滤掉无效价格数据
            valid_highs = [float(row[2]) for row in period_data if row[2] is not None and float(row[2]) > 0]
            valid_lows = [float(row[3]) for row in period_data if row[3] is not None and float(row[3]) > 0]
            valid_volumes = [float(row[5]) for row in period_data if row[5] is not None and float(row[5]) >= 0]
            
            if not valid_highs or not valid_lows:
                continue  # 跳过无效数据
            
            high_price = max(valid_highs)  # 周期内最高价
            low_price = min(valid_lows)  # 周期内最低价
            volume = sum(valid_volumes) if valid_volumes else 0  # 周期内总成交量
            
            # 数据验证
            if high_price < low_price or open_price <= 0 or close_price <= 0:
                logger.warning(f"Invalid OHLC data: O={open_price}, H={high_price}, L={low_price}, C={close_price}")
                continue
            
            aggregated.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
            
        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"Error aggregating period data: {str(e)}")
            continue
    
    logger.debug(f"Successfully aggregated to {len(aggregated)} {timeframe_minutes}min periods")
    return aggregated

def calculate_technical_indicators(ohlcv_data):
    """使用TA-Lib基于OHLCV数据计算专业技术指标"""
    if len(ohlcv_data) < 200:
        logger.warning(f"Insufficient data for technical indicators: {len(ohlcv_data)} periods")
        return None
    
    # 转换为numpy数组
    opens = np.array([float(d['open']) for d in ohlcv_data], dtype=np.float64)
    highs = np.array([float(d['high']) for d in ohlcv_data], dtype=np.float64)
    lows = np.array([float(d['low']) for d in ohlcv_data], dtype=np.float64)
    closes = np.array([float(d['close']) for d in ohlcv_data], dtype=np.float64)
    volumes = np.array([float(d['volume']) for d in ohlcv_data], dtype=np.float64)
    
    try:
        # 计算SMA (Simple Moving Average)
        sma50 = talib.SMA(closes, timeperiod=50)
        sma200 = talib.SMA(closes, timeperiod=200)
        
        # 计算ADX和DMI指标
        adx = talib.ADX(highs, lows, closes, timeperiod=14)
        plus_di = talib.PLUS_DI(highs, lows, closes, timeperiod=14)
        minus_di = talib.MINUS_DI(highs, lows, closes, timeperiod=14)
        
        # 计算布林带
        bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        # 计算布林带宽度（最近20个周期）
        bandwidths = []
        for i in range(len(bb_upper)):
            if not (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_middle[i])):
                if bb_middle[i] != 0:
                    bandwidth = (bb_upper[i] - bb_lower[i]) / bb_middle[i] * 100
                    bandwidths.append(bandwidth)
                else:
                    bandwidths.append(0)
            else:
                bandwidths.append(0)
        
        # 计算ATR (Average True Range)
        atr = talib.ATR(highs, lows, closes, timeperiod=14)
        
        # 获取最新值（处理NaN）
        current_sma50 = sma50[-1] if not np.isnan(sma50[-1]) else closes[-1]
        current_sma200 = sma200[-1] if not np.isnan(sma200[-1]) else closes[-1]
        current_adx = adx[-1] if not np.isnan(adx[-1]) else 0
        current_plus_di = plus_di[-1] if not np.isnan(plus_di[-1]) else 0
        current_minus_di = minus_di[-1] if not np.isnan(minus_di[-1]) else 0
        current_atr = atr[-1] if not np.isnan(atr[-1]) else 0
        
        # 获取最近20个有效的布林带宽度值
        valid_bandwidths = [bw for bw in bandwidths[-20:] if not np.isnan(bw) and bw != 0]
        if not valid_bandwidths:
            valid_bandwidths = [0]
        
        # 获取最近20个有效的ATR值
        valid_atr_values = []
        for atr_val in atr[-20:]:
            if not np.isnan(atr_val):
                valid_atr_values.append(atr_val)
        if not valid_atr_values:
            valid_atr_values = [current_atr]
        
        logger.debug(f"Calculated indicators - SMA50: {current_sma50:.2f}, SMA200: {current_sma200:.2f}, "
                    f"ADX: {current_adx:.2f}, +DI: {current_plus_di:.2f}, -DI: {current_minus_di:.2f}")
        
        return {
            'price': closes[-1],
            'open': ohlcv_data[-1]['open'],
            'high': ohlcv_data[-1]['high'],
            'low': ohlcv_data[-1]['low'],
            'close': closes[-1],
            'volume': ohlcv_data[-1]['volume'],
            'sma50': current_sma50,
            'sma200': current_sma200,
            'adx': current_adx,
            'pdi': current_plus_di,
            'mdi': current_minus_di,
            'bandwidths': valid_bandwidths,
            'atr_values': valid_atr_values
        }
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators with TA-Lib: {str(e)}")
        return None

def analyze_multiple_timeframes(symbol):
    """基于数据库中的5分钟数据分析多个时间框架的趋势"""
    timeframes = {
        "15m": {"minutes": 15, "name": "15分钟"},
        "1h": {"minutes": 60, "name": "1小时"},
        "4h": {"minutes": 240, "name": "4小时"}, 
        "1d": {"minutes": 1440, "name": "1天"},
        "1w": {"minutes": 10080, "name": "1周"}
    }
    
    trends = {}
    insights = {}
    
    for tf, config in timeframes.items():
        try:
            logger.info(f"Calculating {config['name']} indicators from 5min data for {symbol}")
            
            # 基于5分钟数据计算指定时间框架的指标
            indicators = calculate_indicators_from_5min_data(symbol, config['minutes'])
            
            if indicators is None:
                logger.warning(f"Could not calculate indicators for {symbol} {tf}")
                trends[tf] = "数据不足"
                insights[tf] = f"[{config['name']}]数据不足，无法分析趋势"
                continue
            
            # 判断趋势
            trend = determine_trend(indicators, tf)
            insight = generate_insight(symbol, trend, indicators, tf)
            
            trends[tf] = trend
            insights[tf] = insight
            
            # 存储趋势分析
            store_trend_analysis(symbol, tf, trend, insight)
            
            logger.info(f"{symbol} [{config['name']}] 趋势: {trend}")
            
        except Exception as e:
            logger.error(f"Error analyzing {tf} timeframe for {symbol}: {str(e)}")
            trends[tf] = "错误"
            insights[tf] = f"分析{config['name']}趋势时出错: {str(e)}"
    
    return trends, insights

def generate_insight(symbol, trend, indicators=None, timeframe="5m"):
    timeframe_names = {
        "15m": "15分钟", "1h": "1小时", "4h": "4小时", 
        "1d": "1天", "1w": "1周"
    }
    
    tf_name = timeframe_names.get(timeframe, timeframe)
    
    base_insights = {
        "上涨趋势": f"[{tf_name}]基于当前强势ADX和+DI主导，预计继续上涨趋势，可能测试更高阻力位。",
        "下跌趋势": f"[{tf_name}]当前-DI主导且SMA交叉向下，预计延续下跌，关注支撑位。",
        "区间/波动小": f"[{tf_name}]ADX低位且波动率低，预计维持震荡，等待突破信号。"
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
            if abs(sma50 - sma200) / sma200 < 0.02:
                details.append("短长期均线接近，缺乏明确方向")
            
            if details:
                return f"[{tf_name}]当前信号混合：{'; '.join(details)}。建议等待更明确的突破信号。"
        
        return f"[{tf_name}]趋势信号混合，建议观察关键技术位突破情况。"

def store_5min_data(symbol, indicators, interval="5m"):
    """存储5分钟原始数据"""
    try:
        logger.debug(f"Storing 5min data for {symbol}")
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        
        # Create 5min data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crypto_5min_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(20),
                timestamp DATETIME,
                interval_type VARCHAR(10),
                price FLOAT,
                open_price FLOAT,
                high_price FLOAT,
                low_price FLOAT,
                close_price FLOAT,
                volume FLOAT,
                adx FLOAT,
                pdi FLOAT,
                mdi FLOAT,
                sma50 FLOAT,
                sma200 FLOAT,
                bandwidth FLOAT,
                atr FLOAT,
                UNIQUE KEY unique_record (symbol, timestamp, interval_type)
            )
        """)
        
        # Insert 5min data
        now = datetime.now()
        current_bw = indicators['bandwidths'][-1] if 'bandwidths' in indicators else None
        current_atr = indicators['atr_values'][-1] if 'atr_values' in indicators else None
        
        cursor.execute("""
            INSERT INTO crypto_5min_data 
            (symbol, timestamp, interval_type, price, open_price, high_price, low_price, close_price, volume, 
             adx, pdi, mdi, sma50, sma200, bandwidth, atr)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            price=VALUES(price), open_price=VALUES(open_price), high_price=VALUES(high_price),
            low_price=VALUES(low_price), close_price=VALUES(close_price), volume=VALUES(volume),
            adx=VALUES(adx), pdi=VALUES(pdi), mdi=VALUES(mdi), sma50=VALUES(sma50), sma200=VALUES(sma200),
            bandwidth=VALUES(bandwidth), atr=VALUES(atr)
        """, (symbol, now, interval, 
              indicators.get('price', 0), indicators.get('open', 0), indicators.get('high', 0),
              indicators.get('low', 0), indicators.get('close', 0), indicators.get('volume', 0),
              indicators['adx'], indicators['pdi'], indicators['mdi'], 
              indicators['sma50'], indicators['sma200'], current_bw, current_atr))
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.debug(f"Successfully stored 5min data for {symbol}")
    except Exception as e:
        logger.error(f"MySQL error storing 5min data for {symbol}: {str(e)}")
        raise

def store_trend_analysis(symbol, timeframe, trend, insight):
    """存储不同时间框架的趋势分析"""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        
        # Create trends table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crypto_trends (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(20),
                timeframe VARCHAR(10),
                timestamp DATETIME,
                trend VARCHAR(50),
                insight TEXT,
                UNIQUE KEY unique_trend (symbol, timeframe, timestamp)
            )
        """)
        
        now = datetime.now()
        cursor.execute("""
            INSERT INTO crypto_trends (symbol, timeframe, timestamp, trend, insight)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            trend=VALUES(trend), insight=VALUES(insight)
        """, (symbol, timeframe, now, trend, insight))
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.debug(f"Successfully stored trend analysis for {symbol} {timeframe}")
    except Exception as e:
        logger.error(f"MySQL error storing trend for {symbol} {timeframe}: {str(e)}")
        raise

def cleanup_old_data():
    """清理90天前的数据"""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        
        # 清理5分钟数据（保留90天）
        cursor.execute("""
            DELETE FROM crypto_5min_data 
            WHERE timestamp < DATE_SUB(NOW(), INTERVAL 90 DAY)
        """)
        deleted_5min = cursor.rowcount
        
        # 清理趋势数据（保留90天）
        cursor.execute("""
            DELETE FROM crypto_trends 
            WHERE timestamp < DATE_SUB(NOW(), INTERVAL 90 DAY)
        """)
        deleted_trends = cursor.rowcount
        
        conn.commit()
        cursor.close()
        conn.close()
        
        if deleted_5min > 0 or deleted_trends > 0:
            logger.info(f"Cleaned up old data: {deleted_5min} 5min records, {deleted_trends} trend records")
        
    except Exception as e:
        logger.error(f"Error cleaning up old data: {str(e)}")

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
    last_trends = {sym: {tf: None for tf in ["15m", "1h", "4h", "1d", "1w"]} for sym in symbols}
    
    logger.info("Multi-timeframe crypto trend bot started successfully...")
    logger.info("Data strategy: Collect 5min data from API, calculate 15m/1h/4h/1d/1w trends from database")
    logger.info("Monitoring timeframes: 15m, 1h, 4h, 1d, 1w")
    logger.info("Data retention: 90 days, cleanup daily at midnight")
    
    while True:
        try:
            logger.info("Starting new multi-timeframe analysis cycle...")
            all_trends = {}
            all_insights = {}
            trend_changed = False
            
            # 每天清理一次旧数据（在第一次运行时）
            current_hour = datetime.now().hour
            if current_hour == 0:  # 每天午夜清理
                logger.info("Performing daily data cleanup...")
                cleanup_old_data()
            
            for i, symbol in enumerate(symbols):
                logger.info(f"Analyzing {symbol} across multiple timeframes...")
                
                # Add delay between symbols to avoid rate limiting
                if i > 0:
                    logger.info("Waiting 10 seconds to avoid rate limit...")
                    time.sleep(10)
                
                # 首先获取5分钟数据并存储（这是唯一的API调用）
                try:
                    data_5m = fetch_taapi_data(symbol, "5m")
                    indicators_5m = parse_indicators(data_5m)
                    store_5min_data(symbol, indicators_5m, "5m")
                    logger.info(f"Stored 5min data for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to store 5min data for {symbol}: {str(e)}")
                    continue  # 如果无法获取5分钟数据，跳过这个symbol
                
                # 基于数据库中的5分钟数据分析多个时间框架
                symbol_trends, symbol_insights = analyze_multiple_timeframes(symbol)
                all_trends[symbol] = symbol_trends
                all_insights[symbol] = symbol_insights
                
                # 检查是否有趋势变化
                for timeframe in ["15m", "1h", "4h", "1d", "1w"]:
                    current_trend = symbol_trends.get(timeframe, "未知")
                    if current_trend != last_trends[symbol][timeframe]:
                        trend_changed = True
                        logger.info(f"{symbol} [{timeframe}] 趋势变化: {last_trends[symbol][timeframe]} -> {current_trend}")
                        last_trends[symbol][timeframe] = current_trend
                
                # 添加延迟避免API限制
                logger.info("Waiting 15 seconds before next symbol...")
                time.sleep(15)
            
            # 发送通知（如果有趋势变化）
            if trend_changed:
                message = "🔄 加密货币多时间框架趋势更新\n\n"
                
                for symbol in symbols:
                    coin_name = symbol.split('/')[0]
                    message += f"💰 {coin_name}:\n"
                    
                    trends = all_trends[symbol]
                    insights = all_insights[symbol]
                    
                    # 只显示主要时间框架
                    main_timeframes = ["15m", "1h", "4h", "1d", "1w"]
                    for tf in main_timeframes:
                        if tf in trends:
                            tf_name = {"15m": "15分钟", "1h": "1小时", "4h": "4小时", "1d": "1天", "1w": "1周"}[tf]
                            message += f"  {tf_name}: {trends[tf]}\n"
                    
                    # 添加1天的详细分析
                    if "1d" in insights:
                        message += f"  📊 {insights['1d']}\n"
                    
                    message += "\n"
                
                logger.info("Multi-timeframe trend changes detected, sending notification...")
                send_to_telegram(message.strip())
                logger.info("Multi-timeframe notification sent")
            else:
                logger.info("No trend changes detected across all timeframes")
                
            logger.info("Multi-timeframe analysis cycle completed, waiting 5 minut..")
            
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            logger.info("Continuing after error...")
            
        time.sleep(3600)  # 每小时运行一次