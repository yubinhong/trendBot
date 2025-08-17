import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Any

# 导入自定义模块
from database import DatabaseManager
from api_client import BinanceAPIClient
from technical_analysis import TechnicalAnalyzer
from quality_monitor import DataQualityMonitor
from notification import NotificationManager

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

# 初始化组件
db_manager = DatabaseManager(MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB)
api_client = BinanceAPIClient(API_RATE_LIMIT_BUFFER)
technical_analyzer = TechnicalAnalyzer(db_manager, VOLATILITY_THRESHOLD)
quality_monitor = DataQualityMonitor(db_manager, api_client)
notification_manager = NotificationManager()

# Test database connection
try:
    conn = db_manager.get_connection()
    conn.close()
    logger.info("✓ Database connection successful")
except Exception as e:
    logger.error(f"✗ Database connection failed: {str(e)}")
    logger.error("Please check your database configuration and ensure MySQL is running")
    sys.exit(1)

# 确保数据库表存在
db_manager.ensure_database_tables()

# 监控的币种
symbols = ['BTCUSDT', 'ETHUSDT']

# 存储上次的趋势状态
last_trends = {symbol: {} for symbol in symbols}

def send_individual_trend_notification(symbol: str, trends: Dict[str, str], insights: Dict[str, Any], changed_timeframes: List[tuple]):
    """为单个币种发送趋势变化通知"""
    try:
        coin_name = symbol.replace('USDT', '')
        
        # 生成针对单个币种的通知消息
        message = f"🚨 {coin_name} 趋势变化提醒\n"
        message += f"⏰ {datetime.now().strftime('%H:%M')}\n\n"
        
        # 检查数据状态
        data_status = check_data_sufficiency(symbol)
        total_records = data_status.get('total_records', 0)
        days_available = total_records / 288 if total_records > 0 else 0
        
        message += f"💎 {coin_name} 趋势分析\n"
        message += f"📈 数据: {days_available:.1f}天 ({total_records}条)\n\n"
        
        # 趋势状态（使用emoji表示）
        trend_emojis = {
            "上涨趋势": "🟢",
            "下跌趋势": "🔴", 
            "区间/波动小": "🟡",
            "未知": "⚪",
            "数据不足": "⏳",
            "数据积累中": "⏳"
        }
        
        # 显示变化的时间框架
        message += "📊 趋势变化:\n"
        for timeframe, old_trend, new_trend in changed_timeframes:
            tf_name = {"5m": "5分钟", "15m": "15分钟", "1h": "1小时", "4h": "4小时", "1d": "1天"}[timeframe]
            old_emoji = trend_emojis.get(old_trend, "❓")
            new_emoji = trend_emojis.get(new_trend, "❓")
            message += f"  {tf_name}: {old_emoji}{old_trend} → {new_emoji}{new_trend}\n"
        
        message += "\n📈 当前所有时间框架:\n"
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
        
        # 添加数据质量说明
        message += "\nℹ️ 说明:\n"
        message += "🟢上涨 🔴下跌 🟡震荡 ⚪混合 ⏳积累中\n"
        message += "(基础)=数据积累中，分析会持续改善\n"
        message += f"📊 每5分钟更新 | 数据保留250天"
        
        logger.info(f"Sending individual trend notification for {symbol}...")
        notification_manager.send_telegram_message(message.strip())
        logger.info(f"Individual trend notification sent for {symbol}")
        
    except Exception as e:
        logger.error(f"Error sending individual notification for {symbol}: {str(e)}")

def check_data_sufficiency(symbol: str) -> Dict[str, Any]:
    """检查数据充足性"""
    try:
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM crypto_5min_data 
            WHERE symbol = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        """, (symbol,))
        
        total_records = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        # 计算可用天数（每天288个5分钟数据点）
        days_available = total_records / 288 if total_records > 0 else 0
        
        return {
            'total_records': total_records,
            'days_available': days_available,
            'can_analyze_5m': total_records >= 50,
            'can_analyze_15m': total_records >= 100,
            'can_analyze_1h': total_records >= 200,
            'can_analyze_4h': total_records >= 500,
            'can_analyze_1d': total_records >= 2000
        }
        
    except Exception as e:
        logger.error(f"Error checking data sufficiency for {symbol}: {str(e)}")
        return {
            'total_records': 0,
            'days_available': 0,
            'can_analyze_5m': False,
            'can_analyze_15m': False,
            'can_analyze_1h': False,
            'can_analyze_4h': False,
            'can_analyze_1d': False
        }

def analyze_multiple_timeframes(symbol: str) -> tuple[Dict[str, str], Dict[str, str]]:
    """分析多个时间框架"""
    trends = {}
    insights = {}
    
    # 检查数据充足性
    data_status = check_data_sufficiency(symbol)
    
    timeframes = {
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240,
        "1d": 1440
    }
    
    for tf_name, tf_minutes in timeframes.items():
        try:
            # 检查是否有足够数据进行分析
            can_analyze_key = f"can_analyze_{tf_name}"
            if not data_status.get(can_analyze_key, False):
                trends[tf_name] = "数据积累中"
                insights[tf_name] = f"[需要更多数据] 当前: {data_status['total_records']}条记录"
                continue
            
            # 使用技术分析器计算指标
            indicators = technical_analyzer.calculate_indicators_from_5min_data(symbol, tf_minutes)
            
            if indicators:
                trend = technical_analyzer.analyze_trend(indicators, tf_name)
                trends[tf_name] = trend
                
                # 生成洞察
                adx = indicators.get('adx', 0)
                pdi = indicators.get('pdi', 0)
                mdi = indicators.get('mdi', 0)
                
                if adx > 30:
                    if pdi > mdi:
                        insights[tf_name] = f"基于当前强势ADX({adx:.1f})和+DI优势，{trend}信号较强"
                    else:
                        insights[tf_name] = f"基于当前强势ADX({adx:.1f})和-DI优势，{trend}信号较强"
                elif adx < 20:
                    insights[tf_name] = f"ADX低位({adx:.1f})且波动率低，市场处于{trend}状态"
                else:
                    insights[tf_name] = f"信号混合，ADX中等({adx:.1f})，{trend}需要确认"
            else:
                trends[tf_name] = "数据不足"
                insights[tf_name] = "无法计算技术指标"
                
        except Exception as e:
            logger.error(f"Error analyzing {tf_name} timeframe for {symbol}: {str(e)}")
            trends[tf_name] = "分析错误"
            insights[tf_name] = f"分析出错: {str(e)[:50]}"
    
    return trends, insights

def initialize_historical_data(symbol: str, limit: int = 60000) -> bool:
    """初始化历史数据 - 分批获取60000条数据"""
    try:
        logger.info(f"Initializing historical data for {symbol} (limit: {limit})...")
        
        # 计算需要的批次数（每次最多1000条）
        batch_size = 1000
        total_batches = (limit + batch_size - 1) // batch_size  # 向上取整
        
        logger.info(f"Will fetch data in {total_batches} batches of {batch_size} records each")
        
        all_klines = []
        end_time = None  # 从最新数据开始获取
        
        for batch_num in range(total_batches):
            try:
                logger.info(f"Fetching batch {batch_num + 1}/{total_batches} for {symbol}...")
                
                # 获取当前批次的K线数据
                if end_time is None:
                    # 第一批：获取最新的1000条数据
                    klines = api_client.fetch_binance_klines(symbol, '5m', batch_size)
                else:
                    # 后续批次：使用endTime参数获取更早的数据
                    klines = api_client.fetch_binance_klines_with_endtime(symbol, '5m', batch_size, end_time)
                
                if not klines:
                    logger.warning(f"No data returned for batch {batch_num + 1}, stopping...")
                    break
                
                logger.info(f"Retrieved {len(klines)} klines for batch {batch_num + 1}")
                
                # 添加到总数据中
                all_klines.extend(klines)
                
                # 设置下一批次的结束时间（当前批次最早的时间戳）
                if klines:
                    # 获取当前批次最早的时间戳，减1毫秒作为下一批次的endTime
                    # timestamp是datetime对象，需要转换为毫秒时间戳
                    earliest_timestamp = min(int(k['timestamp'].timestamp() * 1000) for k in klines)
                    end_time = earliest_timestamp - 1
                
                # 批次间延迟，避免API限制
                if batch_num < total_batches - 1:  # 不是最后一批
                    logger.info("Waiting 2 seconds before next batch...")
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Error fetching batch {batch_num + 1}: {str(e)}")
                # 继续下一批次，不中断整个过程
                continue
        
        if not all_klines:
            logger.error(f"Failed to fetch any historical klines for {symbol}")
            return False
        
        logger.info(f"Retrieved total {len(all_klines)} historical klines for {symbol}")
        
        # 批量存储历史数据
        success = db_manager.store_historical_klines_bulk(symbol, all_klines, '5m')
        
        if success:
            logger.info(f"Successfully stored {len(all_klines)} historical records for {symbol}")
            return True
        else:
            logger.error(f"Failed to store historical data for {symbol}")
            return False
            
    except Exception as e:
        logger.error(f"Error initializing historical data for {symbol}: {str(e)}")
        return False

def cleanup_old_data():
    """清理旧数据"""
    try:
        db_manager.cleanup_old_data()
        logger.info("Old data cleanup completed")
    except Exception as e:
        logger.error(f"Error during data cleanup: {str(e)}")

def check_expired_trends():
    """检查过期趋势"""
    try:
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM crypto_trends 
            WHERE timestamp < DATE_SUB(NOW(), INTERVAL 7 DAY)
        """)
        
        deleted_count = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} expired trend records")
            
    except Exception as e:
        logger.error(f"Error cleaning expired trends: {str(e)}")

def main():
    """主函数"""
    logger.info("=== Crypto Trend Analysis Bot Started ===")
    
    # 初始化历史数据（如果未跳过）
    if not SKIP_INITIALIZATION:
        logger.info("Starting historical data initialization...")
        initialization_results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Initializing data for {symbol}...")
                
                # 增加到60000条历史数据
                success = initialize_historical_data(symbol, 60000)
                initialization_results[symbol] = success
                
                if success:
                    logger.info(f"✓ Successfully initialized historical data for {symbol}")
                else:
                    logger.warning(f"⚠ Limited initialization for {symbol} - will accumulate data over time")
                
                # 符号间等待5秒（避免Binance接口限制）
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
        elif successful_inits > 0:
            logger.info(f"⚠ Partial initialization success ({successful_inits}/{total_symbols} symbols)")
        else:
            logger.warning("⚠ Historical data initialization failed for all symbols")
    
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
                quality_results = quality_monitor.monitor_data_quality(symbols)
                
                for symbol in symbols:
                    if symbol in quality_results.get('quality_reports', {}):
                        quality_report = quality_results['quality_reports'][symbol]
                        logger.info(f"Data quality report for {symbol}: {quality_report.get('quality_grade', 'F')} ({quality_report.get('overall_score', 0):.1f}%)")
                        
                        # 如果数据质量低，发送警告通知
                        if quality_report.get('overall_score', 0) < 70:
                            notification_manager.send_quality_warning(symbol, quality_report)
                            logger.warning(f"Low data quality alert sent for {symbol}")
            
            for i, symbol in enumerate(symbols):
                logger.info(f"Analyzing {symbol} across multiple timeframes...")
                
                # 每次循环添加5秒延迟（避免Binance接口限制）
                if i > 0:
                    logger.info("Waiting 5 seconds before next symbol...")
                    time.sleep(5)
                
                # 获取最新的5分钟数据并存储
                try:
                    indicators_5m = technical_analyzer.fetch_current_data(symbol)
                    if indicators_5m:
                        db_manager.store_5min_data(symbol, indicators_5m, "5m")
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
                
                # 检查当前币种是否有趋势变化，如有变化立即发送独立通知
                symbol_trend_changed = False
                changed_timeframes = []
                
                for timeframe in ["5m", "15m", "1h", "4h", "1d"]:
                    current_trend = symbol_trends.get(timeframe, "未知")
                    # 初始化last_trends中不存在的时间框架
                    if timeframe not in last_trends[symbol]:
                        last_trends[symbol][timeframe] = "未知"
                    if current_trend != last_trends[symbol][timeframe]:
                        symbol_trend_changed = True
                        trend_changed = True
                        changed_timeframes.append((timeframe, last_trends[symbol][timeframe], current_trend))
                        logger.info(f"{symbol} [{timeframe}] 趋势变化: {last_trends[symbol][timeframe]} -> {current_trend}")
                        last_trends[symbol][timeframe] = current_trend
                
                # 如果当前币种有趋势变化，立即发送独立通知
                if symbol_trend_changed:
                    send_individual_trend_notification(symbol, symbol_trends, symbol_insights, changed_timeframes)
                
                # 添加5秒延迟（避免Binance接口限制）
                logger.info("Waiting 5 seconds before next symbol...")
                time.sleep(5)
            
            # 记录分析周期完成（个别通知已在循环中发送）
            if trend_changed:
                logger.info("Trend changes detected and individual notifications sent")
            else:
                logger.info("No trend changes detected across all timeframes")
                
            logger.info("Multi-timeframe analysis cycle completed, waiting 5 minutes...")
            
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            logger.info("Continuing after error...")
            
        time.sleep(300)  # 每5分钟运行一次

if __name__ == "__main__":
    main()