import requests
import json
import os
import mysql.connector
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import time
import sys
import logging
import hashlib

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
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_DB = os.getenv('MYSQL_DB')
SKIP_INITIALIZATION = os.getenv('SKIP_INITIALIZATION', 'false').lower() == 'true'

# Data granularity configuration
DATA_GRANULARITY = os.getenv('DATA_GRANULARITY', '5m')  # '1m' or '5m'
SMART_GRANULARITY = os.getenv('SMART_GRANULARITY', 'true').lower() == 'true'
VOLATILITY_THRESHOLD = float(os.getenv('VOLATILITY_THRESHOLD', '2.0'))  # ATR multiplier for high volatility
API_RATE_LIMIT_BUFFER = float(os.getenv('API_RATE_LIMIT_BUFFER', '0.8'))  # Use 80% of API limit

if not all([TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB]):
    logger.error("Missing required environment variables.")
    raise ValueError("Missing required environment variables.")

logger.info("Starting crypto trend bot...")
logger.info(f"Monitoring symbols: BTC/USDT, ETH/USDT")
logger.info(f"MySQL Host: {MYSQL_HOST}")
logger.info(f"Telegram Chat ID: {TELEGRAM_CHAT_ID}")
logger.info(f"Data Granularity: {DATA_GRANULARITY}")
logger.info(f"Smart Granularity: {SMART_GRANULARITY}")
logger.info(f"Volatility Threshold: {VOLATILITY_THRESHOLD}")
logger.info(f"API Rate Limit Buffer: {API_RATE_LIMIT_BUFFER}")

# Test database connection
try:
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB
    )
    conn.close()
    logger.info("✓ Database connection successful")
except Exception as e:
    logger.error(f"✗ Database connection failed: {str(e)}")
    logger.error("Please check your database configuration and ensure MySQL is running")
    sys.exit(1)

# API调用频率控制
api_call_count = 0
api_call_reset_time = time.time()
API_WEIGHT_LIMIT = int(1200 * API_RATE_LIMIT_BUFFER)  # 使用配置的缓冲比例

# 智能缓存系统
api_cache = {}
CACHE_EXPIRY_SECONDS = 30  # 缓存30秒
MAX_CACHE_SIZE = 100  # 最大缓存条目数

def reset_api_counter_if_needed():
    """如果需要，重置API调用计数器"""
    global api_call_count, api_call_reset_time
    current_time = time.time()
    if current_time - api_call_reset_time >= 60:  # 每分钟重置
        api_call_count = 0
        api_call_reset_time = current_time
        logger.debug("API call counter reset")

def generate_cache_key(symbol, interval, limit, end_time=None):
    """生成缓存键"""
    key_data = f"{symbol}_{interval}_{limit}_{end_time}"
    return hashlib.md5(key_data.encode()).hexdigest()

def get_from_cache(cache_key):
    """从缓存获取数据"""
    if cache_key in api_cache:
        cached_data, timestamp = api_cache[cache_key]
        if time.time() - timestamp < CACHE_EXPIRY_SECONDS:
            logger.debug(f"Cache hit for key: {cache_key[:8]}...")
            return cached_data
        else:
            # 缓存过期，删除
            del api_cache[cache_key]
            logger.debug(f"Cache expired for key: {cache_key[:8]}...")
    return None

def set_cache(cache_key, data):
    """设置缓存数据"""
    global api_cache
    
    # 如果缓存已满，删除最旧的条目
    if len(api_cache) >= MAX_CACHE_SIZE:
        oldest_key = min(api_cache.keys(), key=lambda k: api_cache[k][1])
        del api_cache[oldest_key]
        logger.debug(f"Cache full, removed oldest entry: {oldest_key[:8]}...")
    
    api_cache[cache_key] = (data, time.time())
    logger.debug(f"Cached data for key: {cache_key[:8]}...")

def cleanup_expired_cache():
    """清理过期的缓存条目"""
    global api_cache
    current_time = time.time()
    expired_keys = []
    
    for key, (data, timestamp) in api_cache.items():
        if current_time - timestamp >= CACHE_EXPIRY_SECONDS:
            expired_keys.append(key)
    
    for key in expired_keys:
        del api_cache[key]
    
    if expired_keys:
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

def check_api_rate_limit(weight=1):
    """检查API调用频率限制"""
    global api_call_count
    reset_api_counter_if_needed()
    
    if api_call_count + weight > API_WEIGHT_LIMIT:
        wait_time = 60 - (time.time() - api_call_reset_time)
        if wait_time > 0:
            logger.warning(f"API rate limit approaching, waiting {wait_time:.1f} seconds")
            time.sleep(wait_time)
            reset_api_counter_if_needed()
    
    api_call_count += weight

def fetch_binance_klines(symbol, interval=None, limit=1000, max_retries=3):
    """从 Binance API 获取 K线数据，支持动态间隔选择和智能缓存"""
    # 如果没有指定间隔，使用智能选择
    if interval is None:
        interval = get_optimal_interval(symbol)
    
    # 生成缓存键并尝试从缓存获取
    cache_key = generate_cache_key(symbol, interval, limit)
    cached_data = get_from_cache(cache_key)
    if cached_data is not None:
        return cached_data
    
    # 转换符号格式：BTC/USDT -> BTCUSDT
    binance_symbol = symbol.replace('/', '')
    
    # 检查API调用频率
    check_api_rate_limit(1)  # K线数据权重为1
    
    # 定期清理过期缓存
    if len(api_cache) > 0 and time.time() % 60 < 1:
        cleanup_expired_cache()
    
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": binance_symbol,
        "interval": interval,
        "limit": limit
    }
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Fetching {limit} {interval} klines for {symbol} (attempt {attempt + 1}/{max_retries})")
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                klines = response.json()
                logger.debug(f"Successfully fetched {len(klines)} klines for {symbol}")
                # 缓存成功的API响应
                set_cache(cache_key, klines)
                return klines
            elif response.status_code == 429:  # Rate limit
                wait_time = (attempt + 1) * 5
                logger.warning(f"Binance rate limit hit for {symbol}, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Binance API error for {symbol}: {response.text}")
                if attempt == max_retries - 1:
                    raise Exception(f"Binance API error for {symbol}: {response.text}")
                time.sleep(2)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {symbol}: {str(e)}")
            if attempt == max_retries - 1:
                raise Exception(f"Request error for {symbol}: {str(e)}")
            time.sleep(2)
    
    raise Exception(f"Failed to fetch klines for {symbol} after {max_retries} attempts")

def fetch_binance_klines_with_endtime(symbol, interval=None, limit=1000, end_time=None, max_retries=3):
    """从 Binance API 获取指定结束时间的 K线数据，支持动态间隔选择和智能缓存"""
    # 如果没有指定间隔，使用智能选择
    if interval is None:
        interval = get_optimal_interval(symbol)
    
    # 生成缓存键并尝试从缓存获取
    cache_key = generate_cache_key(symbol, interval, limit, end_time)
    cached_data = get_from_cache(cache_key)
    if cached_data is not None:
        return cached_data
    
    # 转换符号格式：BTC/USDT -> BTCUSDT
    binance_symbol = symbol.replace('/', '')
    
    # 检查API调用频率
    check_api_rate_limit(1)  # K线数据权重为1
    
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": binance_symbol,
        "interval": interval,
        "limit": limit
    }
    
    if end_time:
        params["endTime"] = end_time
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Fetching {limit} {interval} klines for {symbol} (endTime: {end_time})")
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                klines = response.json()
                logger.debug(f"Successfully fetched {len(klines)} klines for {symbol}")
                # 缓存成功的API响应
                set_cache(cache_key, klines)
                return klines
            elif response.status_code == 429:  # Rate limit
                wait_time = (attempt + 1) * 5
                logger.warning(f"Binance rate limit hit for {symbol}, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Binance API error for {symbol}: {response.text}")
                if attempt == max_retries - 1:
                    raise Exception(f"Binance API error for {symbol}: {response.text}")
                time.sleep(2)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {symbol}: {str(e)}")
            if attempt == max_retries - 1:
                raise Exception(f"Request error for {symbol}: {str(e)}")
            time.sleep(2)
    
    raise Exception(f"Failed to fetch klines for {symbol} after {max_retries} attempts")

def convert_klines_to_ohlcv(klines):
    """将 Binance K线数据转换为 OHLCV 格式"""
    ohlcv_data = []
    
    for kline in klines:
        try:
            # Binance kline format: [timestamp, open, high, low, close, volume, ...]
            timestamp = datetime.fromtimestamp(int(kline[0]) / 1000)  # 毫秒转秒
            open_price = float(kline[1])
            high_price = float(kline[2])
            low_price = float(kline[3])
            close_price = float(kline[4])
            volume = float(kline[5])
            
            ohlcv_data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing kline data: {str(e)}")
            continue
    
    return ohlcv_data

def detect_high_volatility(symbol, lookback_periods=20):
    """检测当前市场波动率，判断是否需要更细粒度数据"""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        
        # 获取最近的ATR数据
        cursor.execute("""
            SELECT atr, close_price
            FROM crypto_5min_data 
            WHERE symbol = %s AND atr > 0
            ORDER BY timestamp DESC
            LIMIT %s
        """, (symbol, lookback_periods))
        
        data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if len(data) < 10:  # 需要足够的数据点
            logger.debug(f"Insufficient ATR data for volatility detection: {len(data)} records")
            return False
        
        # 计算ATR相对于价格的比率
        atr_values = [row[0] for row in data]
        prices = [row[1] for row in data]
        
        current_atr = atr_values[0]
        current_price = prices[0]
        avg_atr = np.mean(atr_values)
        
        # 计算ATR相对于价格的百分比
        atr_percentage = (current_atr / current_price) * 100
        avg_atr_percentage = (avg_atr / np.mean(prices)) * 100
        
        # 判断是否为高波动
        volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1
        is_high_volatility = volatility_ratio > VOLATILITY_THRESHOLD
        
        logger.debug(f"{symbol} volatility analysis: ATR={current_atr:.4f}, Ratio={volatility_ratio:.2f}, High={is_high_volatility}")
        
        return is_high_volatility
        
    except Exception as e:
        logger.error(f"Error detecting volatility for {symbol}: {str(e)}")
        return False

def get_optimal_interval(symbol):
    """根据配置和市场条件确定最优的数据间隔"""
    # 如果禁用智能粒度，直接返回配置的粒度
    if not SMART_GRANULARITY:
        return DATA_GRANULARITY
    
    # 如果配置为1分钟，直接返回
    if DATA_GRANULARITY == '1m':
        return '1m'
    
    # 检测高波动率
    if detect_high_volatility(symbol):
        logger.info(f"{symbol} high volatility detected, using 1m data")
        return '1m'
    else:
        logger.debug(f"{symbol} normal volatility, using {DATA_GRANULARITY} data")
        return DATA_GRANULARITY

def aggregate_1min_to_5min(df_1min):
    """将1分钟数据聚合为5分钟数据"""
    try:
        if df_1min.empty:
            return pd.DataFrame()
        
        # 确保时间戳是datetime类型
        df_1min['timestamp'] = pd.to_datetime(df_1min['timestamp'])
        df_1min.set_index('timestamp', inplace=True)
        
        # 按5分钟重采样
        df_5min = df_1min.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # 重置索引
        df_5min.reset_index(inplace=True)
        
        logger.debug(f"Aggregated {len(df_1min)} 1min records to {len(df_5min)} 5min records")
        return df_5min
        
    except Exception as e:
        logger.error(f"Error aggregating 1min to 5min data: {str(e)}")
        return pd.DataFrame()

def fetch_current_data(symbol):
    """获取最新数据并计算技术指标，支持动态粒度选择"""
    try:
        # 获取最优间隔
        optimal_interval = get_optimal_interval(symbol)
        
        # 根据间隔调整获取的数据量
        if optimal_interval == '1m':
            limit = 1500  # 1分钟数据需要更多点数来覆盖相同时间范围
        else:
            limit = 300   # 5分钟数据
        
        # 获取最新的K线数据
        klines = fetch_binance_klines(symbol, optimal_interval, limit)
        if not klines:
            logger.error(f"No klines data received for {symbol}")
            return None
        
        # 转换为OHLCV格式
        ohlcv_data = convert_klines_to_ohlcv(klines)
        
        # 如果获取的是1分钟数据，需要聚合成5分钟数据用于存储
        if optimal_interval == '1m':
            # 聚合1分钟数据为5分钟数据
            ohlcv_5min = aggregate_1min_to_5min(ohlcv_data.copy())
            if ohlcv_5min.empty:
                logger.error(f"Failed to aggregate 1m to 5m data for {symbol}")
                return None
            # 使用聚合后的5分钟数据进行存储和分析
            ohlcv_data = ohlcv_5min
        
        if len(ohlcv_data) < 200:
            logger.warning(f"Insufficient klines data for {symbol}: {len(ohlcv_data)} records")
            return None
        
        # 计算技术指标
        indicators = calculate_technical_indicators(ohlcv_data)
        
        if indicators is None:
            logger.error(f"Failed to calculate indicators for {symbol}")
            return None
        
        logger.debug(f"Successfully processed 5min data for {symbol}")
        return indicators
        
    except Exception as e:
        logger.error(f"Error fetching current 5min data for {symbol}: {str(e)}")
        return None



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
    
    if timeframe == "5m":
        adx_strong_threshold = 40  # 5分钟需要非常强的信号
        adx_moderate_threshold = 35
        sma_diff_threshold = 0.005
    elif timeframe == "15m":
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
        
        # 根据时间框架调整最小数据要求
        min_periods_needed = {
            15: 20,    # 15分钟需要20个周期 (约5小时)
            60: 25,    # 1小时需要25个周期 (约1天)  
            240: 15,   # 4小时需要15个周期 (约2.5天) - 基础分析
            1440: 200  # 1天需要200个周期 (约200天) - 保持高准确度
        }.get(timeframe_minutes, 20)
        
        if len(raw_data) < periods_per_timeframe * min_periods_needed:
            logger.warning(f"Insufficient 5min data for {symbol} {timeframe_minutes}m analysis: {len(raw_data)} records (need {periods_per_timeframe * min_periods_needed})")
            return None
        
        logger.debug(f"Retrieved {len(raw_data)} 5min records for {symbol}")
        
        # 将5分钟数据聚合为目标时间框架
        aggregated_data = aggregate_5min_to_timeframe(raw_data, timeframe_minutes)
        
        # 根据时间框架调整聚合数据的最小要求
        min_aggregated_periods = {
            15: 20,    # 15分钟需要20个聚合周期
            60: 25,    # 1小时需要25个聚合周期
            240: 15,   # 4小时需要15个聚合周期（基础分析）
            1440: 200  # 1天需要200个聚合周期（保持高准确度）
        }.get(timeframe_minutes, 20)
        
        if len(aggregated_data) < min_aggregated_periods:
            logger.warning(f"Insufficient aggregated data for {symbol} {timeframe_minutes}m: {len(aggregated_data)} periods (need {min_aggregated_periods})")
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
    """使用pandas-ta基于OHLCV数据计算专业技术指标，支持高波动期增强处理"""
    # 动态调整最小数据要求，确保能计算基本指标
    min_periods = max(50, len(ohlcv_data) // 4)  # 至少50个周期，或数据的1/4
    
    # 根据数据量动态调整要求
    if len(ohlcv_data) >= 200:
        # 数据充足，可以进行完整分析
        min_required = 200
    elif len(ohlcv_data) >= 50:
        # 数据适中，可以进行基础分析（主要用于4小时）
        min_required = 50
    else:
        # 数据不足，无法分析
        logger.warning(f"Insufficient data for technical indicators: {len(ohlcv_data)} periods (need at least 50)")
        return None
    
    try:
        # 转换为pandas DataFrame
        df = pd.DataFrame(ohlcv_data)
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # 删除包含NaN的行
        df = df.dropna()
        
        if len(df) < 200:
            logger.warning(f"Insufficient clean data after removing NaN: {len(df)} periods")
            return None
        
        # 计算SMA (Simple Moving Average)
        df['sma50'] = ta.sma(df['close'], length=50)
        df['sma200'] = ta.sma(df['close'], length=200)
        
        # 计算ADX和DMI指标
        adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_data['ADX_14']
        df['plus_di'] = adx_data['DMP_14']
        df['minus_di'] = adx_data['DMN_14']
        
        # 计算布林带
        bbands = ta.bbands(df['close'], length=20, std=2)
        df['bb_upper'] = bbands['BBU_20_2.0']
        df['bb_middle'] = bbands['BBM_20_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0']
        
        # 计算布林带宽度
        df['bandwidth'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100).fillna(0)
        
        # 计算ATR (Average True Range)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # 高波动期增强处理：检测当前波动率水平
        recent_atr = df['atr'].tail(20).dropna()
        if len(recent_atr) >= 10:
            current_atr_value = recent_atr.iloc[-1]
            avg_atr = recent_atr.mean()
            volatility_ratio = current_atr_value / avg_atr if avg_atr > 0 else 1
            is_high_volatility = volatility_ratio > VOLATILITY_THRESHOLD
            
            # 在高波动期动态调整指标参数
            if is_high_volatility:
                logger.debug(f"High volatility detected (ratio: {volatility_ratio:.2f}), using enhanced indicators")
                
                # 使用更短周期的ADX以更快响应变化
                adx_data_short = ta.adx(df['high'], df['low'], df['close'], length=10)
                df['adx_enhanced'] = adx_data_short['ADX_10']
                df['plus_di_enhanced'] = adx_data_short['DMP_10']
                df['minus_di_enhanced'] = adx_data_short['DMN_10']
                
                # 计算额外的波动率指标
                df['rsi'] = ta.rsi(df['close'], length=14)  # RSI用于超买超卖判断
                df['stoch_k'] = ta.stoch(df['high'], df['low'], df['close'])['STOCHk_14_3_3']  # 随机指标
                
                # 使用增强的指标值
                use_enhanced = True
            else:
                use_enhanced = False
        else:
            use_enhanced = False
            is_high_volatility = False
        
        # 获取最新值（处理NaN）
        current_sma50 = df['sma50'].iloc[-1] if not pd.isna(df['sma50'].iloc[-1]) else df['close'].iloc[-1]
        current_sma200 = df['sma200'].iloc[-1] if not pd.isna(df['sma200'].iloc[-1]) else df['close'].iloc[-1]
        
        # 根据波动率选择ADX指标
        if use_enhanced and 'adx_enhanced' in df.columns:
            current_adx = df['adx_enhanced'].iloc[-1] if not pd.isna(df['adx_enhanced'].iloc[-1]) else 0
            current_plus_di = df['plus_di_enhanced'].iloc[-1] if not pd.isna(df['plus_di_enhanced'].iloc[-1]) else 0
            current_minus_di = df['minus_di_enhanced'].iloc[-1] if not pd.isna(df['minus_di_enhanced'].iloc[-1]) else 0
        else:
            current_adx = df['adx'].iloc[-1] if not pd.isna(df['adx'].iloc[-1]) else 0
            current_plus_di = df['plus_di'].iloc[-1] if not pd.isna(df['plus_di'].iloc[-1]) else 0
            current_minus_di = df['minus_di'].iloc[-1] if not pd.isna(df['minus_di'].iloc[-1]) else 0
        
        current_atr = df['atr'].iloc[-1] if not pd.isna(df['atr'].iloc[-1]) else 0
        
        # 获取最近20个有效的布林带宽度值
        valid_bandwidths = df['bandwidth'].tail(20).dropna().tolist()
        if not valid_bandwidths:
            valid_bandwidths = [0]
        
        # 获取最近20个有效的ATR值
        valid_atr_values = df['atr'].tail(20).dropna().tolist()
        if not valid_atr_values:
            valid_atr_values = [current_atr]
        
        # 获取额外的高波动期指标
        current_rsi = None
        current_stoch = None
        if use_enhanced:
            current_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1]) else None
            current_stoch = df['stoch_k'].iloc[-1] if 'stoch_k' in df.columns and not pd.isna(df['stoch_k'].iloc[-1]) else None
        
        logger.debug(f"Calculated indicators - SMA50: {current_sma50:.2f}, SMA200: {current_sma200:.2f}, "
                    f"ADX: {current_adx:.2f}, +DI: {current_plus_di:.2f}, -DI: {current_minus_di:.2f}")
        
        if is_high_volatility:
            logger.debug(f"High volatility indicators - RSI: {current_rsi}, Stoch: {current_stoch}")
        
        result = {
            'price': df['close'].iloc[-1],
            'open': df['open'].iloc[-1],
            'high': df['high'].iloc[-1],
            'low': df['low'].iloc[-1],
            'close': df['close'].iloc[-1],
            'volume': df['volume'].iloc[-1],
            'sma50': current_sma50,
            'sma200': current_sma200,
            'adx': current_adx,
            'pdi': current_plus_di,
            'mdi': current_minus_di,
            'bandwidths': valid_bandwidths,
            'atr_values': valid_atr_values,
            'is_high_volatility': is_high_volatility,
            'volatility_ratio': volatility_ratio if 'volatility_ratio' in locals() else 1.0
        }
        
        # 添加高波动期的额外指标
        if use_enhanced:
            result.update({
                'rsi': current_rsi,
                'stoch_k': current_stoch,
                'enhanced_mode': True
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators with pandas-ta: {str(e)}")
        return None

def check_data_sufficiency(symbol):
    """检查数据是否足够进行分析"""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        
        # 检查最近7天的数据量
        cursor.execute("""
            SELECT COUNT(*) FROM crypto_5min_data 
            WHERE symbol = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        """, (symbol,))
        
        recent_count = cursor.fetchone()[0]
        
        # 检查总数据量
        cursor.execute("""
            SELECT COUNT(*) FROM crypto_5min_data WHERE symbol = %s
        """, (symbol,))
        
        total_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        # 评估数据充足性（4小时基础分析，1天高准确度分析）
        sufficiency = {
            "total_records": total_count,
            "recent_records": recent_count,
            "can_analyze_5m": total_count >= 20,      # 需要至少20个5分钟数据点
            "can_analyze_15m": total_count >= 60,     # 需要至少5小时数据
            "can_analyze_1h": total_count >= 240,     # 需要至少20小时数据  
            "can_analyze_4h": total_count >= 720,     # 需要至少2.5天数据（基础分析）
            "can_analyze_1d": total_count >= 57600,   # 需要至少200天数据（完整分析）
        }
        
        return sufficiency
        
    except Exception as e:
        logger.error(f"Error checking data sufficiency for {symbol}: {str(e)}")
        return None

def analyze_multiple_timeframes(symbol):
    """基于数据库中的5分钟数据分析多个时间框架的趋势"""
    timeframes = {
        "5m": {"minutes": 5, "name": "5分钟"},
        "15m": {"minutes": 15, "name": "15分钟"},
        "1h": {"minutes": 60, "name": "1小时"},
        "4h": {"minutes": 240, "name": "4小时"}, 
        "1d": {"minutes": 1440, "name": "1天"}
    }
    
    # 检查数据充足性
    data_sufficiency = check_data_sufficiency(symbol)
    if data_sufficiency:
        logger.info(f"{symbol} data status: {data_sufficiency['total_records']} total, {data_sufficiency['recent_records']} recent")
    
    trends = {}
    insights = {}
    
    for tf, config in timeframes.items():
        try:
            # 检查该时间框架是否有足够数据
            can_analyze_key = f"can_analyze_{tf}"
            if data_sufficiency and not data_sufficiency.get(can_analyze_key, False):
                logger.warning(f"Insufficient data for {symbol} {tf} analysis")
                trends[tf] = "数据积累中"
                
                if tf == "1d":
                    # 1天趋势需要特别说明
                    total_records = data_sufficiency.get('total_records', 0)
                    days_available = total_records / 288  # 每天288个5分钟数据点
                    insights[tf] = f"[{config['name']}]数据积累中（当前约{days_available:.1f}天数据），完整分析需要200天数据。基础分析将随数据积累逐步改善。"
                else:
                    insights[tf] = f"[{config['name']}]数据积累中，请等待更多数据收集后再分析"
                continue
            
            logger.info(f"Calculating {config['name']} indicators from 5min data for {symbol}")
            
            # 基于5分钟数据计算指定时间框架的指标
            indicators = calculate_indicators_from_5min_data(symbol, config['minutes'])
            
            if indicators is None:
                logger.warning(f"Could not calculate indicators for {symbol} {tf}")
                trends[tf] = "数据不足"
                insights[tf] = f"[{config['name']}]数据不足，无法分析趋势。建议等待更多数据积累。"
                continue
            
            # 判断趋势
            trend = determine_trend(indicators, tf)
            insight = generate_insight(symbol, trend, indicators, tf)
            
            trends[tf] = trend
            insights[tf] = insight
            
            # 存储趋势分析（包含ADX强度用于有效期计算）
            adx_strength = indicators.get('adx', 0) if indicators else 0
            store_trend_analysis(symbol, tf, trend, insight, adx_strength)
            
            logger.info(f"{symbol} [{config['name']}] 趋势: {trend}")
            
        except Exception as e:
            logger.error(f"Error analyzing {tf} timeframe for {symbol}: {str(e)}")
            trends[tf] = "错误"
            insights[tf] = f"分析{config['name']}趋势时出错: {str(e)}"
    
    return trends, insights

def get_trend_validity_period(timeframe, adx_strength):
    """估算趋势预测的有效期"""
    base_periods = {
        "5m": {"min": 10, "max": 30, "unit": "分钟"},
        "15m": {"min": 1, "max": 4, "unit": "小时"},
        "1h": {"min": 4, "max": 24, "unit": "小时"},
        "4h": {"min": 1, "max": 7, "unit": "天"},
        "1d": {"min": 1, "max": 4, "unit": "周"}
    }
    
    if timeframe not in base_periods:
        return "未知"
    
    period = base_periods[timeframe]
    
    # 根据ADX强度调整有效期
    if adx_strength > 30:  # 强趋势
        validity = f"{period['max']}{period['unit']}"
        confidence = "高"
    elif adx_strength > 20:  # 中等趋势
        mid_period = (period['min'] + period['max']) // 2
        validity = f"{mid_period}{period['unit']}"
        confidence = "中等"
    else:  # 弱趋势
        validity = f"{period['min']}{period['unit']}"
        confidence = "低"
    
    return f"预期有效期{validity}(置信度:{confidence})"

def generate_insight(symbol, trend, indicators=None, timeframe="5m"):
    timeframe_names = {
        "5m": "5分钟", "15m": "15分钟", "1h": "1小时", "4h": "4小时", 
        "1d": "1天"
    }
    
    tf_name = timeframe_names.get(timeframe, timeframe)
    
    # 获取ADX强度用于有效期估算
    adx_strength = indicators.get('adx', 0) if indicators else 0
    validity_info = get_trend_validity_period(timeframe, adx_strength)
    
    # 添加风险提醒
    risk_warning = ""
    if timeframe == "5m":
        risk_warning = "⚠️超短期趋势波动大，仅供参考，不建议单独作为交易依据"
    elif timeframe in ["15m", "1h"]:
        risk_warning = "⚠️短期趋势易受突发事件影响"
    elif adx_strength < 20:
        risk_warning = "⚠️趋势强度较弱，注意反转风险"
    
    base_insights = {
        "上涨趋势": f"强势上涨信号，ADX和+DI主导，可能测试更高阻力位。{validity_info}。{risk_warning}",
        "下跌趋势": f"下跌信号确认，-DI主导且均线向下，关注支撑位。{validity_info}。{risk_warning}",
        "区间/波动小": f"低波动震荡，ADX较低，等待方向性突破。{validity_info}。{risk_warning}"
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
                return f"[{tf_name}]当前信号混合：{'; '.join(details)}。{validity_info}。建议等待更明确的突破信号。"
        
        return f"[{tf_name}]趋势信号混合，建议观察关键技术位突破情况。{validity_info}。"

def store_1min_data(symbol, ohlcv_data, interval="1m"):
    """存储1分钟原始数据"""
    try:
        logger.debug(f"Storing 1min data for {symbol}")
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        
        # 批量插入1分钟数据
        stored_count = 0
        for record in ohlcv_data:
            try:
                cursor.execute("""
                    INSERT INTO crypto_1min_data 
                    (symbol, timestamp, interval_type, price, open_price, high_price, low_price, close_price, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    price=VALUES(price), open_price=VALUES(open_price), high_price=VALUES(high_price),
                    low_price=VALUES(low_price), close_price=VALUES(close_price), volume=VALUES(volume)
                """, (symbol, record['timestamp'], interval,
                      record['close'], record['open'], record['high'],
                      record['low'], record['close'], record['volume']))
                
                if cursor.rowcount > 0:
                    stored_count += 1
                    
            except Exception as e:
                logger.warning(f"Error storing 1min record: {str(e)}")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.debug(f"Successfully stored {stored_count} 1min records for {symbol}")
        return stored_count
        
    except Exception as e:
        logger.error(f"MySQL error storing 1min data for {symbol}: {str(e)}")
        raise

def store_5min_data(symbol, indicators, interval="5m", data_source="api_direct"):
    """存储5分钟数据和技术指标，支持增强字段"""
    try:
        logger.debug(f"Storing 5min data for {symbol} (source: {data_source})")
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        
        # Insert 5min data with enhanced fields
        now = datetime.now()
        current_bw = indicators['bandwidths'][-1] if 'bandwidths' in indicators else None
        current_atr = indicators['atr_values'][-1] if 'atr_values' in indicators else None
        
        # 提取增强模式的指标
        rsi = indicators.get('rsi')
        stoch_k = indicators.get('stoch_k')
        is_high_volatility = indicators.get('is_high_volatility', False)
        volatility_ratio = indicators.get('volatility_ratio')
        enhanced_mode = indicators.get('enhanced_mode', False)
        
        cursor.execute("""
            INSERT INTO crypto_5min_data 
            (symbol, timestamp, interval_type, price, open_price, high_price, low_price, close_price, volume, 
             adx, pdi, mdi, sma50, sma200, bandwidth, atr, rsi, stoch_k, is_high_volatility, 
             volatility_ratio, enhanced_mode, data_source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            price=VALUES(price), open_price=VALUES(open_price), high_price=VALUES(high_price),
            low_price=VALUES(low_price), close_price=VALUES(close_price), volume=VALUES(volume),
            adx=VALUES(adx), pdi=VALUES(pdi), mdi=VALUES(mdi), sma50=VALUES(sma50), sma200=VALUES(sma200),
            bandwidth=VALUES(bandwidth), atr=VALUES(atr), rsi=VALUES(rsi), stoch_k=VALUES(stoch_k),
            is_high_volatility=VALUES(is_high_volatility), volatility_ratio=VALUES(volatility_ratio),
            enhanced_mode=VALUES(enhanced_mode), data_source=VALUES(data_source)
        """, (symbol, now, interval, 
              indicators.get('price', 0), indicators.get('open', 0), indicators.get('high', 0),
              indicators.get('low', 0), indicators.get('close', 0), indicators.get('volume', 0),
              indicators['adx'], indicators['pdi'], indicators['mdi'], 
              indicators['sma50'], indicators['sma200'], current_bw, current_atr,
              rsi, stoch_k, is_high_volatility, volatility_ratio, enhanced_mode, data_source))
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.debug(f"Successfully stored 5min data for {symbol}")
    except Exception as e:
        logger.error(f"MySQL error storing 5min data for {symbol}: {str(e)}")
        raise

def store_trend_analysis(symbol, timeframe, trend, insight, adx_strength=0):
    """存储不同时间框架的趋势分析，包含有效期信息"""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        
        # 计算预期有效期（小时）
        validity_hours = {
            "5m": 0.5 if adx_strength > 30 else 0.25,  # 30分钟 vs 15分钟
            "15m": 2 if adx_strength > 30 else 1,
            "1h": 12 if adx_strength > 30 else 6,
            "4h": 72 if adx_strength > 30 else 24,
            "1d": 336 if adx_strength > 30 else 168  # 2周 vs 1周
        }.get(timeframe, 24)
        
        now = datetime.now()
        cursor.execute("""
            INSERT INTO crypto_trends (symbol, timeframe, timestamp, trend, insight, adx_strength, expected_validity_hours)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            trend=VALUES(trend), insight=VALUES(insight), adx_strength=VALUES(adx_strength), 
            expected_validity_hours=VALUES(expected_validity_hours), is_expired=FALSE
        """, (symbol, timeframe, now, trend, insight, adx_strength, validity_hours))
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.debug(f"Successfully stored trend analysis for {symbol} {timeframe} (validity: {validity_hours}h)")
    except Exception as e:
        logger.error(f"MySQL error storing trend for {symbol} {timeframe}: {str(e)}")
        raise

def check_expired_trends():
    """检查并标记过期的趋势预测"""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        
        # 标记过期的趋势
        cursor.execute("""
            UPDATE crypto_trends 
            SET is_expired = TRUE 
            WHERE is_expired = FALSE 
            AND TIMESTAMPDIFF(HOUR, timestamp, NOW()) > expected_validity_hours
        """)
        
        expired_count = cursor.rowcount
        
        conn.commit()
        cursor.close()
        conn.close()
        
        if expired_count > 0:
            logger.info(f"Marked {expired_count} trend predictions as expired")
            
    except Exception as e:
        logger.error(f"Error checking expired trends: {str(e)}")

def ensure_database_tables():
    """确保数据库表存在，支持多粒度数据存储"""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        
        # 创建1分钟数据表（原始高精度数据）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crypto_1min_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(20),
                timestamp DATETIME,
                interval_type VARCHAR(10) DEFAULT '1m',
                price FLOAT,
                open_price FLOAT,
                high_price FLOAT,
                low_price FLOAT,
                close_price FLOAT,
                volume FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_record (symbol, timestamp, interval_type),
                INDEX idx_symbol_timestamp (symbol, timestamp),
                INDEX idx_timestamp (timestamp)
            )
        """)
        
        # 创建5分钟数据表（聚合数据和技术指标）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crypto_5min_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(20),
                timestamp DATETIME,
                interval_type VARCHAR(10) DEFAULT '5m',
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
                rsi FLOAT,
                stoch_k FLOAT,
                is_high_volatility BOOLEAN DEFAULT FALSE,
                volatility_ratio FLOAT,
                enhanced_mode BOOLEAN DEFAULT FALSE,
                data_source ENUM('api_direct', 'aggregated_1m') DEFAULT 'api_direct',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_record (symbol, timestamp, interval_type),
                INDEX idx_symbol_timestamp (symbol, timestamp),
                INDEX idx_timestamp (timestamp),
                INDEX idx_volatility (is_high_volatility)
            )
        """)
        
        # 创建趋势分析表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crypto_trends (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(20),
                timeframe VARCHAR(10),
                timestamp DATETIME,
                trend VARCHAR(50),
                insight TEXT,
                adx_strength FLOAT,
                expected_validity_hours INT,
                is_expired BOOLEAN DEFAULT FALSE,
                data_quality ENUM('high', 'medium', 'low') DEFAULT 'medium',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_trend (symbol, timeframe, timestamp),
                INDEX idx_symbol_timeframe (symbol, timeframe),
                INDEX idx_timestamp (timestamp)
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database tables created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        return False

def initialize_historical_data(symbol):
    """初始化时获取足够的历史数据"""
    try:
        logger.info(f"Initializing historical data for {symbol}...")
        
        # 首先确保表存在
        if not ensure_database_tables():
            logger.error("Failed to create database tables")
            return False
        
        # 检查现有数据量
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM crypto_5min_data 
            WHERE symbol = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        """, (symbol,))
        
        existing_count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        # 如果已有一些数据，可以选择跳过初始化
        if existing_count >= 100:  # 降低阈值，有100条记录就足够开始分析
            logger.info(f"{symbol} already has sufficient data ({existing_count} records)")
            return True
        elif existing_count > 0:
            logger.info(f"{symbol} has {existing_count} records, will try to fetch more historical data")
        else:
            logger.info(f"{symbol} has no historical data, attempting to fetch initial data")
        
        # 只获取5分钟数据，然后基于这些数据聚合计算其他时间框架
        # 1天趋势的SMA200需要200天数据 = 57,600个5分钟数据点
        # 考虑到初始化时间，我们采用渐进式策略：
        # 1. 先获取足够的数据支持短期分析（15分钟、1小时、4小时）
        # 2. 1天趋势会随着时间积累逐步改善
        
        # 分批获取5分钟数据
        # 目标：获取约30天的数据，足够支持4小时趋势的完整分析
        batches_needed = 9  # 9批 × 1000 = 9,000条记录（约31天）
        
        logger.info(f"Will fetch {batches_needed} batches of 5min data (total ~9000 records, ~31 days)")
        logger.info("This supports full analysis for 15m/1h/4h trends. 1-day trend will improve over time.")
        
        intervals_to_fetch = [("5m", 1000)] * batches_needed
        
        total_stored = 0
        
        # 获取历史数据的策略：
        # 1. 第一批获取最新的1000条5分钟数据
        # 2. 后续批次使用最早数据的时间戳作为endTime，继续向前获取
        
        last_timestamp = None
        
        for batch_num in range(1, len(intervals_to_fetch) + 1):
            try:
                interval, limit = intervals_to_fetch[0]  # 都是5分钟数据
                logger.info(f"Fetching batch {batch_num}/{len(intervals_to_fetch)}: {limit} {interval} klines for {symbol}")
                
                # 获取历史K线数据
                if batch_num == 1:
                    # 第一批：获取最新数据
                    ohlcv_data = fetch_historical_klines_bulk(symbol, interval, limit)
                else:
                    # 后续批次：使用endTime获取更早的数据
                    if last_timestamp:
                        end_time = int(last_timestamp.timestamp() * 1000)  # 转换为毫秒
                        ohlcv_data = fetch_historical_klines_with_time(symbol, interval, limit, end_time)
                    else:
                        logger.warning(f"Batch {batch_num}: no last_timestamp available, skipping")
                        continue
                
                if ohlcv_data and len(ohlcv_data) > 0:
                    # 存储数据
                    stored_count = store_historical_klines_bulk(symbol, ohlcv_data, interval)
                    total_stored += stored_count
                    
                    # 更新最早时间戳用于下一批
                    timestamps = [record['timestamp'] for record in ohlcv_data]
                    last_timestamp = min(timestamps)
                    
                    logger.info(f"Batch {batch_num}: fetched {len(ohlcv_data)} records, stored {stored_count}, earliest: {last_timestamp}")
                    
                    # 如果存储的数据少于获取的数据，说明有重复数据，可能需要停止
                    if stored_count == 0:
                        logger.info(f"Batch {batch_num}: no new records stored (likely duplicates), stopping")
                        break
                else:
                    logger.warning(f"Batch {batch_num}: no klines data received")
                    break  # 没有更多数据了，停止获取
                
                # 批次间短暂等待
                if batch_num < len(intervals_to_fetch):
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching batch {batch_num} for {symbol}: {str(e)}")
                continue
        
        if total_stored == 0:
            logger.warning(f"No historical data could be fetched for {symbol} due to API limits")
            logger.info(f"System will start with current data and accumulate over time")
            logger.info(f"This is normal with strict API limits - the system will work fine!")
        else:
            logger.info(f"Successfully initialized {total_stored} historical records for {symbol}")
        
        logger.info(f"Historical data initialization completed for {symbol}: {total_stored} total records")
        # 即使没有获取到历史数据，也返回True，因为系统可以正常运行
        return True
        
    except Exception as e:
        logger.error(f"Error initializing historical data for {symbol}: {str(e)}")
        return False

def fetch_historical_klines_bulk(symbol, interval, limit):
    """批量获取历史K线数据"""
    try:
        klines = fetch_binance_klines(symbol, interval, limit)
        if klines:
            return convert_klines_to_ohlcv(klines)
        return None
    except Exception as e:
        logger.error(f"Error fetching historical klines for {symbol} {interval}: {str(e)}")
        return None

def fetch_historical_klines_with_time(symbol, interval, limit, end_time):
    """获取指定结束时间的历史K线数据"""
    try:
        klines = fetch_binance_klines_with_endtime(symbol, interval, limit, end_time)
        if klines:
            return convert_klines_to_ohlcv(klines)
        return None
    except Exception as e:
        logger.error(f"Error fetching historical klines with endTime for {symbol} {interval}: {str(e)}")
        return None

def store_historical_klines_bulk(symbol, ohlcv_data, interval):
    """批量存储历史K线数据"""
    if not ohlcv_data:
        return 0
    
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        
        stored_count = 0
        
        # 如果是1分钟数据，存储到1分钟表
        if interval == "1m":
            for record in ohlcv_data:
                try:
                    cursor.execute("""
                        INSERT IGNORE INTO crypto_1min_data 
                        (symbol, timestamp, interval_type, price, open_price, high_price, low_price, close_price, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        symbol, record['timestamp'], "1m",
                        record['close'],  # price
                        record['open'],
                        record['high'],
                        record['low'],
                        record['close'],
                        record['volume']
                    ))
                    
                    if cursor.rowcount > 0:
                        stored_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error storing 1min kline record: {str(e)}")
                    continue
        
        # 如果是5分钟数据，直接存储到5分钟表
        elif interval == "5m":
            for record in ohlcv_data:
                try:
                    cursor.execute("""
                        INSERT IGNORE INTO crypto_5min_data 
                        (symbol, timestamp, interval_type, price, open_price, high_price, low_price, close_price, volume, 
                         adx, pdi, mdi, sma50, sma200, bandwidth, atr, rsi, stoch_k, is_high_volatility, 
                         volatility_ratio, enhanced_mode, data_source)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        symbol, record['timestamp'], "5m",
                        record['close'],  # price
                        record['open'],
                        record['high'],
                        record['low'],
                        record['close'],
                        record['volume'],
                        0, 0, 0, 0, 0, 0, 0, 0, 0, False, 0, False, 'api_direct'  # 技术指标稍后计算
                    ))
                    
                    if cursor.rowcount > 0:
                        stored_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error storing kline record: {str(e)}")
                    continue
        else:
            # 对于其他时间间隔，分解为5分钟数据点
            interval_minutes = {
                "15m": 15, "1h": 60, "4h": 240, "1d": 1440
            }.get(interval, 5)
            
            points_per_interval = max(1, interval_minutes // 5)
            
            for record in ohlcv_data:
                try:
                    base_timestamp = record['timestamp']
                    
                    for i in range(points_per_interval):
                        point_timestamp = base_timestamp + pd.Timedelta(minutes=i*5)
                        
                        cursor.execute("""
                            INSERT IGNORE INTO crypto_5min_data 
                            (symbol, timestamp, interval_type, price, open_price, high_price, low_price, close_price, volume, 
                             adx, pdi, mdi, sma50, sma200, bandwidth, atr)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            symbol, point_timestamp, "5m",
                            record['close'],  # price
                            record['open'],
                            record['high'],
                            record['low'],
                            record['close'],
                            record['volume'],
                            0, 0, 0, 0, 0, 0, 0  # 技术指标稍后计算
                        ))
                        
                        if cursor.rowcount > 0:
                            stored_count += 1
                            
                except Exception as e:
                    logger.warning(f"Error storing expanded kline record: {str(e)}")
                    continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return stored_count
        
    except Exception as e:
        logger.error(f"Error storing historical klines bulk: {str(e)}")
        return 0

def monitor_data_quality(symbol):
    """监控数据质量和精度对比"""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        
        quality_report = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'data_completeness': {},
            'precision_comparison': {},
            'anomaly_detection': {},
            'overall_quality': 'unknown'
        }
        
        # 1. 检查数据完整性
        # 检查1分钟数据完整性（最近1小时）
        cursor.execute("""
            SELECT COUNT(*) as actual_count,
                   60 as expected_count,
                   (COUNT(*) / 60.0 * 100) as completeness_pct
            FROM crypto_1min_data 
            WHERE symbol = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
        """, (symbol,))
        
        result = cursor.fetchone()
        if result:
            quality_report['data_completeness']['1min'] = {
                'actual_count': result[0],
                'expected_count': result[1],
                'completeness_pct': round(result[2], 2)
            }
        
        # 检查5分钟数据完整性（最近1小时）
        cursor.execute("""
            SELECT COUNT(*) as actual_count,
                   12 as expected_count,
                   (COUNT(*) / 12.0 * 100) as completeness_pct
            FROM crypto_5min_data 
            WHERE symbol = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
        """, (symbol,))
        
        result = cursor.fetchone()
        if result:
            quality_report['data_completeness']['5min'] = {
                'actual_count': result[0],
                'expected_count': result[1],
                'completeness_pct': round(result[2], 2)
            }
        
        # 2. 精度对比：比较聚合数据与直接API数据的差异
        cursor.execute("""
            SELECT 
                AVG(CASE WHEN data_source = 'api_direct' THEN close_price END) as api_direct_avg,
                AVG(CASE WHEN data_source = 'aggregated_1m' THEN close_price END) as aggregated_avg,
                COUNT(CASE WHEN data_source = 'api_direct' THEN 1 END) as direct_count,
                COUNT(CASE WHEN data_source = 'aggregated_1m' THEN 1 END) as aggregated_count
            FROM crypto_5min_data 
            WHERE symbol = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
        """, (symbol,))
        
        result = cursor.fetchone()
        if result and result[0] and result[1]:
            price_diff_pct = abs((result[0] - result[1]) / result[0] * 100)
            quality_report['precision_comparison'] = {
                'api_direct_avg_price': round(result[0], 4),
                'aggregated_avg_price': round(result[1], 4),
                'price_diff_pct': round(price_diff_pct, 4),
                'direct_count': result[2],
                'aggregated_count': result[3]
            }
        
        # 3. 异常检测：检查价格跳跃和数据异常
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN ABS((close_price - LAG(close_price) OVER (ORDER BY timestamp)) / LAG(close_price) OVER (ORDER BY timestamp)) > 0.05 THEN 1 END) as large_jumps,
                COUNT(*) as total_records,
                AVG(volatility_ratio) as avg_volatility_ratio,
                COUNT(CASE WHEN is_high_volatility = 1 THEN 1 END) as high_volatility_periods
            FROM crypto_5min_data 
            WHERE symbol = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
        """, (symbol,))
        
        result = cursor.fetchone()
        if result:
            jump_rate = (result[0] / max(result[1], 1)) * 100 if result[1] > 0 else 0
            quality_report['anomaly_detection'] = {
                'large_price_jumps': result[0],
                'total_records': result[1],
                'jump_rate_pct': round(jump_rate, 2),
                'avg_volatility_ratio': round(result[2] or 0, 2),
                'high_volatility_periods': result[3]
            }
        
        # 4. 计算整体质量评分
        quality_score = 0
        max_score = 0
        
        # 数据完整性评分 (40%)
        if '1min' in quality_report['data_completeness']:
            completeness_1m = quality_report['data_completeness']['1min']['completeness_pct']
            quality_score += (completeness_1m / 100) * 20
        max_score += 20
        
        if '5min' in quality_report['data_completeness']:
            completeness_5m = quality_report['data_completeness']['5min']['completeness_pct']
            quality_score += (completeness_5m / 100) * 20
        max_score += 20
        
        # 精度评分 (30%)
        if quality_report['precision_comparison']:
            price_diff = quality_report['precision_comparison']['price_diff_pct']
            precision_score = max(0, (1 - price_diff / 5) * 30)  # 5%差异为0分
            quality_score += precision_score
        max_score += 30
        
        # 异常检测评分 (30%)
        if quality_report['anomaly_detection']:
            jump_rate = quality_report['anomaly_detection']['jump_rate_pct']
            anomaly_score = max(0, (1 - jump_rate / 10) * 30)  # 10%跳跃率为0分
            quality_score += anomaly_score
        max_score += 30
        
        # 计算最终质量等级
        if max_score > 0:
            final_score = (quality_score / max_score) * 100
            if final_score >= 90:
                quality_report['overall_quality'] = 'high'
            elif final_score >= 70:
                quality_report['overall_quality'] = 'medium'
            else:
                quality_report['overall_quality'] = 'low'
            
            quality_report['quality_score'] = round(final_score, 2)
        
        cursor.close()
        conn.close()
        
        # 记录质量报告
        logger.info(f"Data quality report for {symbol}: {quality_report['overall_quality']} ({quality_report.get('quality_score', 0):.1f}%)")
        
        # 如果质量较低，记录警告
        if quality_report['overall_quality'] == 'low':
            logger.warning(f"Low data quality detected for {symbol}. Consider investigating data sources.")
        
        return quality_report
        
    except Exception as e:
        logger.error(f"Error monitoring data quality for {symbol}: {str(e)}")
        return None

def cleanup_old_data():
    """清理旧数据，保留足够支持1天趋势分析的数据"""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        
        # 清理1分钟数据（保留7天，用于高波动期分析）
        cursor.execute("""
            DELETE FROM crypto_1min_data 
            WHERE timestamp < DATE_SUB(NOW(), INTERVAL 7 DAY)
        """)
        deleted_1min = cursor.rowcount
        
        # 清理5分钟数据（保留250天，确保1天趋势SMA200有足够数据）
        cursor.execute("""
            DELETE FROM crypto_5min_data 
            WHERE timestamp < DATE_SUB(NOW(), INTERVAL 250 DAY)
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
        
        if deleted_1min > 0 or deleted_5min > 0 or deleted_trends > 0:
            logger.info(f"Cleaned up old data: {deleted_1min} 1min records (>7 days), {deleted_5min} 5min records (>250 days), {deleted_trends} trend records (>90 days)")
        
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
    last_trends = {sym: {tf: None for tf in ["5m", "15m", "1h", "4h", "1d"]} for sym in symbols}
    
    logger.info("Multi-timeframe crypto trend bot started successfully...")
    logger.info("Data strategy: Collect 5min data from API, calculate 5m/15m/1h/4h/1d trends from database")
    logger.info("Monitoring timeframes: 5m, 15m, 1h, 4h, 1d")
    logger.info("Data retention: 250 days for 5min data, 90 days for trends, cleanup daily at midnight")
    
    # 首先确保数据库表存在
    logger.info("Setting up database tables...")
    if not ensure_database_tables():
        logger.error("Failed to create database tables. Exiting...")
        sys.exit(1)
    
    # 初始化历史数据
    if SKIP_INITIALIZATION:
        logger.info("Skipping historical data initialization (SKIP_INITIALIZATION=true)")
        logger.info("System will start immediately and accumulate data over time")
        initialization_results = {symbol: True for symbol in symbols}
    else:
        logger.info("Checking and initializing historical data...")
        logger.info("Note: Due to API rate limits, initialization may take several minutes")
        logger.info("Set SKIP_INITIALIZATION=true in .env to skip this step")
        
        initialization_results = {}
    
        for symbol in symbols:
            try:
                logger.info(f"Starting initialization for {symbol}...")
                success = initialize_historical_data(symbol)
                initialization_results[symbol] = success
                
                if success:
                    logger.info(f"✓ Successfully initialized historical data for {symbol}")
                else:
                    logger.warning(f"⚠ Limited initialization for {symbol} - will accumulate data over time")
                
                # 符号间短暂等待（Binance API限制宽松）
                if symbol != symbols[-1]:  # 不是最后一个符号
                    logger.info("Waiting 5 seconds before next symbol...")
                    time.sleep(5)
                    
            except Exception as e:
                logger.error(f"Error during initialization for {symbol}: {str(e)}")
                initialization_results[symbol] = True  # 设为True，让系统继续运行
    
    successful_inits = sum(1 for success in initialization_results.values() if success)
    total_symbols = len(symbols)
    
    if successful_inits == total_symbols:
        logger.info("✓ Historical data initialization completed successfully for all symbols")
        logger.info("All timeframes should be available for analysis")
    elif successful_inits > 0:
        logger.info(f"⚠ Partial initialization success ({successful_inits}/{total_symbols} symbols)")
        logger.info("System will start with mixed analysis capabilities")
    else:
        logger.warning("⚠ Historical data initialization failed for all symbols")
        logger.info("System will start with minimal capabilities and accumulate data over time")
        logger.info("This is normal when API rate limits are strict - analysis will improve over time")
    
    # 显示当前数据状态
    for symbol in symbols:
        try:
            sufficiency = check_data_sufficiency(symbol)
            if sufficiency:
                available_timeframes = []
                for tf in ["5m", "15m", "1h", "4h", "1d"]:
                    if sufficiency.get(f"can_analyze_{tf}", False):
                        available_timeframes.append(tf)
                
                if available_timeframes:
                    logger.info(f"{symbol} - Available timeframes: {', '.join(available_timeframes)}")
                else:
                    logger.info(f"{symbol} - No timeframes ready yet, will accumulate data")
        except Exception as e:
            logger.error(f"Error checking initial data status for {symbol}: {str(e)}")
    
    logger.info("Starting main monitoring loop...")
    
    while True:
        try:
            logger.info("Starting new multi-timeframe analysis cycle...")
            all_trends = {}
            all_insights = {}
            trend_changed = False
            
            # 每天清理一次旧数据和检查过期趋势
            current_hour = datetime.now().hour
            if current_hour == 0:  # 每天午夜清理
                logger.info("Performing daily data cleanup...")
                cleanup_old_data()
                check_expired_trends()
                
                # 执行数据质量监控
                logger.info("Performing daily data quality monitoring...")
                for symbol in symbols:
                    quality_report = monitor_data_quality(symbol)
                    if quality_report:
                        logger.info(f"Data quality report for {symbol}: {quality_report['overall_quality']} ({quality_report.get('quality_score', 0):.1f}%)")
                        
                        # 如果数据质量低，发送警告通知
                        if quality_report['overall_quality'] == 'low':
                            warning_message = f"⚠️ 数据质量警告\n\n"
                            warning_message += f"币种: {symbol}\n"
                            warning_message += f"质量等级: {quality_report['overall_quality']}\n"
                            warning_message += f"质量评分: {quality_report.get('quality_score', 0):.1f}%\n\n"
                            
                            if 'data_completeness' in quality_report:
                                warning_message += "数据完整性:\n"
                                for timeframe, data in quality_report['data_completeness'].items():
                                    warning_message += f"  {timeframe}: {data['completeness_pct']:.1f}%\n"
                            
                            if 'anomaly_detection' in quality_report:
                                anomaly = quality_report['anomaly_detection']
                                warning_message += f"\n异常检测:\n"
                                warning_message += f"  价格跳跃率: {anomaly.get('jump_rate_pct', 0):.1f}%\n"
                                warning_message += f"  高波动期: {anomaly.get('high_volatility_periods', 0)}次\n"
                            
                            warning_message += "\n建议检查数据源和网络连接。"
                            send_to_telegram(warning_message)
                            logger.warning(f"Low data quality alert sent for {symbol}")
            
            for i, symbol in enumerate(symbols):
                logger.info(f"Analyzing {symbol} across multiple timeframes...")
                
                # Add minimal delay between symbols (Binance API is more generous)
                if i > 0:
                    logger.info("Waiting 2 seconds before next symbol...")
                    time.sleep(2)
                
                # 获取最新的5分钟数据并存储
                try:
                    indicators_5m = fetch_current_data(symbol)
                    if indicators_5m:
                        store_5min_data(symbol, indicators_5m, "5m")
                        logger.info(f"Stored latest 5min data for {symbol}")
                    else:
                        logger.error(f"Failed to get 5min data for {symbol}")
                        continue  # 如果无法获取5分钟数据，跳过这个symbol
                except Exception as e:
                    logger.error(f"Failed to process 5min data for {symbol}: {str(e)}")
                    continue
                
                # 基于数据库中的5分钟数据分析多个时间框架
                symbol_trends, symbol_insights = analyze_multiple_timeframes(symbol)
                all_trends[symbol] = symbol_trends
                all_insights[symbol] = symbol_insights
                
                # 检查是否有趋势变化
                for timeframe in ["5m", "15m", "1h", "4h", "1d"]:
                    current_trend = symbol_trends.get(timeframe, "未知")
                    # 初始化last_trends中不存在的时间框架
                    if timeframe not in last_trends[symbol]:
                        last_trends[symbol][timeframe] = "未知"
                    if current_trend != last_trends[symbol][timeframe]:
                        trend_changed = True
                        logger.info(f"{symbol} [{timeframe}] 趋势变化: {last_trends[symbol][timeframe]} -> {current_trend}")
                        last_trends[symbol][timeframe] = current_trend
                
                # 添加短暂延迟（Binance API限制宽松）
                logger.info("Waiting 3 seconds before next symbol...")
                time.sleep(3)
            
            # 发送通知（如果有趋势变化）
            if trend_changed:
                # 生成优化的通知消息
                message = "🚨 趋势变化提醒\n"
                message += f"⏰ {datetime.now().strftime('%H:%M')}\n\n"
                
                for symbol in symbols:
                    coin_name = symbol.split('/')[0]
                    trends = all_trends[symbol]
                    insights = all_insights[symbol]
                    
                    # 检查数据状态
                    data_status = check_data_sufficiency(symbol)
                    total_records = data_status.get('total_records', 0) if data_status else 0
                    days_available = total_records / 288 if total_records > 0 else 0
                    
                    # 币种标题
                    message += f"💎 {coin_name} 趋势分析\n"
                    message += f"📈 数据: {days_available:.1f}天 ({total_records}条)\n"
                    
                    # 趋势状态（使用emoji表示）
                    trend_emojis = {
                        "上涨趋势": "🟢",
                        "下跌趋势": "🔴", 
                        "区间/波动小": "🟡",
                        "未知": "⚪",
                        "数据不足": "⏳",
                        "数据积累中": "⏳"
                    }
                    
                    main_timeframes = ["5m", "15m", "1h", "4h", "1d"]
                    for tf in main_timeframes:
                        if tf in trends:
                            tf_name = {"5m": "5分钟", "15m": "15分钟", "1h": "1小时", "4h": "4小时", "1d": "1天"}[tf]
                            trend = trends[tf]
                            emoji = trend_emojis.get(trend, "❓")
                            
                            # 添加分析质量标识
                            quality_indicator = ""
                            if tf == "4h" and days_available < 30:
                                quality_indicator = " (基础)"
                            elif tf == "1d" and days_available < 200:
                                quality_indicator = " (基础)"
                            
                            message += f"  {emoji} {tf_name}: {trend}{quality_indicator}\n"
                    
                    # 添加关键洞察（选择最重要的时间框架）
                    key_insight = None
                    for tf in ["1d", "4h", "1h", "15m", "5m"]:  # 优先级顺序
                        if tf in insights and not insights[tf].startswith("["):
                            # 简化洞察文本
                            insight_text = insights[tf]
                            if "基于当前强势ADX" in insight_text:
                                key_insight = f"💡 强势{trends.get(tf, '未知')}信号"
                            elif "ADX低位且波动率低" in insight_text:
                                key_insight = "💡 低波动震荡，等待突破"
                            elif "信号混合" in insight_text:
                                key_insight = "💡 信号混合，观察关键位"
                            break
                    
                    if key_insight:
                        message += f"  {key_insight}\n"
                    
                    message += "\n"
                
                # 添加数据质量说明
                message += "ℹ️ 说明:\n"
                message += "🟢上涨 🔴下跌 🟡震荡 ⚪混合 ⏳积累中\n"
                message += "(基础)=数据积累中，分析会持续改善\n"
                message += f"📊 每5分钟更新 | 数据保留250天"
                
                logger.info("Multi-timeframe trend changes detected, sending notification...")
                send_to_telegram(message.strip())
                logger.info("Multi-timeframe notification sent")
            else:
                logger.info("No trend changes detected across all timeframes")
                
            logger.info("Multi-timeframe analysis cycle completed, waiting 5 minut..")
            
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            logger.info("Continuing after error...")
            
        time.sleep(300)  # 每5分钟运行一次