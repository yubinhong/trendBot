import mysql.connector
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class DatabaseManager:
    """数据库管理类"""
    
    def __init__(self, host: str, user: str, password: str, database: str):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
    
    def get_connection(self):
        """获取数据库连接"""
        return mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )
    
    def ensure_database_tables(self):
        """确保数据库表存在，支持多粒度数据存储"""
        try:
            conn = self.get_connection()
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
                    trend_direction ENUM('上涨', '下跌', '震荡', '未知') DEFAULT '未知',
                    confidence_score FLOAT DEFAULT 0,
                    support_level FLOAT,
                    resistance_level FLOAT,
                    key_insights TEXT,
                    prediction_accuracy FLOAT,
                    market_sentiment ENUM('极度恐惧', '恐惧', '中性', '贪婪', '极度贪婪') DEFAULT '中性',
                    volatility_level ENUM('低', '中', '高', '极高') DEFAULT '中',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    UNIQUE KEY unique_trend (symbol, timeframe, created_at),
                    INDEX idx_symbol_timeframe (symbol, timeframe),
                    INDEX idx_active_trends (is_active, expires_at)
                )
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("✓ Database tables ensured")
            
        except Exception as e:
            logger.error(f"Error ensuring database tables: {str(e)}")
            raise
    
    def store_1min_data(self, symbol: str, ohlcv_data: List[Dict], interval: str = "1m") -> int:
        """存储1分钟原始数据"""
        try:
            logger.debug(f"Storing 1min data for {symbol}")
            conn = self.get_connection()
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
    
    def store_5min_data(self, symbol: str, indicators: Dict[str, Any], interval: str = "5m", data_source: str = "api_direct"):
        """存储5分钟数据和技术指标，支持增强字段"""
        try:
            logger.debug(f"Storing 5min data for {symbol} (source: {data_source})")
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Insert 5min data with enhanced fields
            now = datetime.now()
            current_bw = indicators['bandwidths'][-1] if 'bandwidths' in indicators else None
            current_atr = indicators['atr_values'][-1] if 'atr_values' in indicators else None
            
            # 提取增强模式的指标，确保类型转换
            rsi = indicators.get('rsi')
            stoch_k = indicators.get('stoch_k')
            is_high_volatility = bool(indicators.get('is_high_volatility', False))  # 转换为Python bool
            volatility_ratio = indicators.get('volatility_ratio')
            enhanced_mode = bool(indicators.get('enhanced_mode', False))  # 转换为Python bool
            
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
    
    def store_trend_analysis(self, symbol: str, timeframe: str, trend_data: Dict[str, Any]):
        """存储趋势分析结果"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 计算过期时间（根据时间框架）
            timeframe_hours = {
                "5m": 0.5, "15m": 1, "1h": 4, "4h": 12, "1d": 48
            }
            hours = timeframe_hours.get(timeframe, 4)
            expires_at = datetime.now() + pd.Timedelta(hours=hours)
            
            cursor.execute("""
                INSERT INTO crypto_trends 
                (symbol, timeframe, trend_direction, confidence_score, support_level, resistance_level, 
                 key_insights, prediction_accuracy, market_sentiment, volatility_level, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                symbol, timeframe, trend_data.get('direction', '未知'),
                trend_data.get('confidence', 0), trend_data.get('support', 0),
                trend_data.get('resistance', 0), trend_data.get('insights', ''),
                trend_data.get('accuracy', 0), trend_data.get('sentiment', '中性'),
                trend_data.get('volatility', '中'), expires_at
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing trend analysis for {symbol}: {str(e)}")
            raise
    
    def cleanup_old_data(self):
        """清理过期数据"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 清理1分钟数据（保留7天）
            cursor.execute("""
                DELETE FROM crypto_1min_data 
                WHERE created_at < DATE_SUB(NOW(), INTERVAL 7 DAY)
            """)
            deleted_1min = cursor.rowcount
            
            # 清理5分钟数据（保留250天）
            cursor.execute("""
                DELETE FROM crypto_5min_data 
                WHERE created_at < DATE_SUB(NOW(), INTERVAL 250 DAY)
            """)
            deleted_5min = cursor.rowcount
            
            # 清理趋势数据（保留90天）
            cursor.execute("""
                DELETE FROM crypto_trends 
                WHERE created_at < DATE_SUB(NOW(), INTERVAL 90 DAY)
            """)
            deleted_trends = cursor.rowcount
            
            conn.commit()
            cursor.close()
            conn.close()
            
            if deleted_1min > 0 or deleted_5min > 0 or deleted_trends > 0:
                logger.info(f"Cleaned up old data: {deleted_1min} 1min, {deleted_5min} 5min, {deleted_trends} trends")
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
    
    def get_historical_data(self, symbol: str, timeframe_minutes: int = 5, limit: int = 1000) -> List[tuple]:
        """从数据库获取历史数据"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if timeframe_minutes == 1:
                table = "crypto_1min_data"
            else:
                table = "crypto_5min_data"
            
            cursor.execute(f"""
                SELECT timestamp, open_price, high_price, low_price, close_price, volume
                FROM {table}
                WHERE symbol = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (symbol, limit))
            
            data = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return []
    
    def store_historical_klines_bulk(self, symbol: str, ohlcv_data: List[Dict], interval: str) -> int:
        """批量存储历史K线数据"""
        if not ohlcv_data:
            return 0
        
        try:
            conn = self.get_connection()
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
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Stored {stored_count} historical {interval} records for {symbol}")
            return stored_count
            
        except Exception as e:
            logger.error(f"Error storing historical klines for {symbol}: {str(e)}")
            return 0