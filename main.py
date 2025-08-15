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
    """使用pandas-ta基于OHLCV数据计算专业技术指标"""
    if len(ohlcv_data) < 200:
        logger.warning(f"Insufficient data for technical indicators: {len(ohlcv_data)} periods")
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
        
        # 获取最新值（处理NaN）
        current_sma50 = df['sma50'].iloc[-1] if not pd.isna(df['sma50'].iloc[-1]) else df['close'].iloc[-1]
        current_sma200 = df['sma200'].iloc[-1] if not pd.isna(df['sma200'].iloc[-1]) else df['close'].iloc[-1]
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
        
        logger.debug(f"Calculated indicators - SMA50: {current_sma50:.2f}, SMA200: {current_sma200:.2f}, "
                    f"ADX: {current_adx:.2f}, +DI: {current_plus_di:.2f}, -DI: {current_minus_di:.2f}")
        
        return {
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
            'atr_values': valid_atr_values
        }
        
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
        
        # 评估数据充足性
        sufficiency = {
            "total_records": total_count,
            "recent_records": recent_count,
            "can_analyze_15m": recent_count >= 60,    # 需要至少5小时数据
            "can_analyze_1h": recent_count >= 240,    # 需要至少20小时数据
            "can_analyze_4h": total_count >= 1000,    # 需要至少3-4天数据
            "can_analyze_1d": total_count >= 2000,    # 需要至少7天数据
            "can_analyze_1w": total_count >= 10000    # 需要至少35天数据
        }
        
        return sufficiency
        
    except Exception as e:
        logger.error(f"Error checking data sufficiency for {symbol}: {str(e)}")
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
        "15m": {"min": 1, "max": 4, "unit": "小时"},
        "1h": {"min": 4, "max": 24, "unit": "小时"},
        "4h": {"min": 1, "max": 7, "unit": "天"},
        "1d": {"min": 1, "max": 4, "unit": "周"},
        "1w": {"min": 1, "max": 6, "unit": "个月"}
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
        "15m": "15分钟", "1h": "1小时", "4h": "4小时", 
        "1d": "1天", "1w": "1周"
    }
    
    tf_name = timeframe_names.get(timeframe, timeframe)
    
    # 获取ADX强度用于有效期估算
    adx_strength = indicators.get('adx', 0) if indicators else 0
    validity_info = get_trend_validity_period(timeframe, adx_strength)
    
    # 添加风险提醒
    risk_warning = ""
    if timeframe in ["15m", "1h"]:
        risk_warning = "⚠️短期趋势易受突发事件影响"
    elif adx_strength < 20:
        risk_warning = "⚠️趋势强度较弱，注意反转风险"
    
    base_insights = {
        "上涨趋势": f"[{tf_name}]基于当前强势ADX和+DI主导，预计继续上涨趋势，可能测试更高阻力位。{validity_info}。{risk_warning}",
        "下跌趋势": f"[{tf_name}]当前-DI主导且SMA交叉向下，预计延续下跌，关注支撑位。{validity_info}。{risk_warning}",
        "区间/波动小": f"[{tf_name}]ADX低位且波动率低，预计维持震荡，等待突破信号。{validity_info}。{risk_warning}"
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
            "15m": 2 if adx_strength > 30 else 1,
            "1h": 12 if adx_strength > 30 else 6,
            "4h": 72 if adx_strength > 30 else 24,
            "1d": 336 if adx_strength > 30 else 168,  # 2周 vs 1周
            "1w": 2160 if adx_strength > 30 else 720   # 3个月 vs 1个月
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
    """确保数据库表存在"""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        
        # 创建5分钟数据表
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
                UNIQUE KEY unique_trend (symbol, timeframe, timestamp)
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
        
        # 如果已有足够数据（7天约2016个5分钟数据点），跳过初始化
        if existing_count >= 1500:
            logger.info(f"{symbol} already has sufficient data ({existing_count} records)")
            return True
        
        logger.info(f"{symbol} has {existing_count} records, need to fetch historical data")
        
        # 获取不同时间间隔的历史数据来快速填充（受API限制，每次最多20个结果）
        intervals_to_fetch = [
            ("5m", 20),    # 最近20个5分钟数据
            ("15m", 20),   # 最近20个15分钟数据  
            ("1h", 20),    # 最近20个1小时数据
            ("4h", 20),    # 最近20个4小时数据
            ("1d", 20),    # 最近20个1天数据
        ]
        
        total_stored = 0
        
        for interval, batch_size in intervals_to_fetch:
            try:
                logger.info(f"Fetching {batch_size} {interval} historical data for {symbol}")
                
                # 使用多次小批量请求来获取更多数据
                batches_to_fetch = 3  # 获取3批，总共60个数据点
                
                for batch_num in range(batches_to_fetch):
                    try:
                        logger.debug(f"Fetching batch {batch_num + 1}/{batches_to_fetch} for {interval}")
                        
                        # 获取历史数据
                        historical_data = fetch_historical_data_bulk(symbol, interval, batch_size)
                        
                        if historical_data:
                            # 转换并存储数据
                            stored_count = store_historical_data_bulk(symbol, historical_data, interval, batch_num)
                            total_stored += stored_count
                            logger.debug(f"Stored {stored_count} {interval} records (batch {batch_num + 1})")
                        
                        # 避免API限制 - 批次间等待更长时间
                        if batch_num < batches_to_fetch - 1:
                            time.sleep(10)
                            
                    except Exception as e:
                        logger.warning(f"Error fetching batch {batch_num + 1} of {interval} data: {str(e)}")
                        # 如果遇到速率限制，等待更长时间
                        if "rate-limit" in str(e).lower():
                            logger.info("Rate limit hit, waiting 30 seconds...")
                            time.sleep(30)
                        continue
                
                logger.info(f"Completed fetching {interval} data for {symbol}")
                
                # 时间间隔间等待
                time.sleep(15)
                
            except Exception as e:
                logger.error(f"Error fetching {interval} data for {symbol}: {str(e)}")
                continue
        
        if total_stored == 0:
            logger.warning(f"No historical data could be fetched for {symbol} due to API limits")
            logger.info(f"System will start with current data and accumulate over time")
        
        logger.info(f"Historical data initialization completed for {symbol}: {total_stored} total records")
        return total_stored > 0
        
    except Exception as e:
        logger.error(f"Error initializing historical data for {symbol}: {str(e)}")
        return False

def fetch_historical_data_bulk(symbol, interval, limit):
    """批量获取历史数据"""
    url = "https://api.taapi.io/bulk"
    payload = {
        "secret": TAAPI_KEY,
        "construct": {
            "exchange": "binance",
            "symbol": symbol,
            "interval": interval,
            "indicators": [
                {"indicator": "price", "results": limit, "id": "price_history"}
            ]
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()['data']
            for item in data:
                if item['id'] == 'price_history':
                    return item['result']
        else:
            logger.error(f"API error fetching {interval} history for {symbol}: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Request error fetching {interval} history for {symbol}: {str(e)}")
        return None

def store_historical_data_bulk(symbol, historical_data, interval, batch_offset=0):
    """批量存储历史数据"""
    if not historical_data:
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
        
        # 将不同时间间隔的数据转换为5分钟数据点
        for record in historical_data:
            try:
                # 解析时间戳
                if 'timestamp' in record:
                    timestamp = datetime.fromtimestamp(record['timestamp'])
                else:
                    continue
                
                # 根据时间间隔创建多个5分钟数据点
                interval_minutes = {
                    "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440
                }.get(interval, 5)
                
                # 为了快速填充数据，我们将较大时间间隔的数据分解为多个5分钟点
                points_per_interval = max(1, interval_minutes // 5)
                
                # 添加批次偏移，避免重复时间戳
                base_offset_minutes = batch_offset * interval_minutes
                
                for i in range(points_per_interval):
                    point_timestamp = timestamp - pd.Timedelta(minutes=base_offset_minutes + i*5)
                    
                    cursor.execute("""
                        INSERT IGNORE INTO crypto_5min_data 
                        (symbol, timestamp, interval_type, price, open_price, high_price, low_price, close_price, volume, 
                         adx, pdi, mdi, sma50, sma200, bandwidth, atr)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        symbol, point_timestamp, "5m",
                        record.get('value', 0),
                        record.get('open', record.get('value', 0)),
                        record.get('high', record.get('value', 0)),
                        record.get('low', record.get('value', 0)),
                        record.get('value', 0),  # close
                        record.get('volume', 0),
                        0, 0, 0, 0, 0, 0, 0  # 技术指标稍后计算
                    ))
                    
                    if cursor.rowcount > 0:
                        stored_count += 1
                        
            except Exception as e:
                logger.warning(f"Error storing historical record: {str(e)}")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return stored_count
        
    except Exception as e:
        logger.error(f"Error storing historical data bulk: {str(e)}")
        return 0

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
    
    # 首先确保数据库表存在
    logger.info("Setting up database tables...")
    if not ensure_database_tables():
        logger.error("Failed to create database tables. Exiting...")
        sys.exit(1)
    
    # 初始化历史数据
    logger.info("Checking and initializing historical data...")
    logger.info("Note: Due to API rate limits, initialization may take several minutes")
    
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
            
            # 符号间等待更长时间避免API限制
            if symbol != symbols[-1]:  # 不是最后一个符号
                logger.info("Waiting 30 seconds before next symbol to avoid API limits...")
                time.sleep(30)
                
        except Exception as e:
            logger.error(f"Error during initialization for {symbol}: {str(e)}")
            initialization_results[symbol] = False
    
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
                for tf in ["15m", "1h", "4h", "1d", "1w"]:
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
            
        time.sleep(300)  # 每5分钟运行一次