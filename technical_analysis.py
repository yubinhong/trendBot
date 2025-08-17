import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from database import DatabaseManager

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """技术分析器"""
    
    def __init__(self, db_manager: DatabaseManager, volatility_threshold: float = 2.0):
        self.db_manager = db_manager
        self.volatility_threshold = volatility_threshold
    
    def detect_high_volatility(self, symbol: str, lookback_periods: int = 20) -> bool:
        """检测当前市场波动率，判断是否需要更细粒度数据"""
        try:
            conn = self.db_manager.get_connection()
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
            is_high_volatility = volatility_ratio > self.volatility_threshold
            
            logger.debug(f"{symbol} volatility analysis: ATR={current_atr:.4f}, Ratio={volatility_ratio:.2f}, High={is_high_volatility}")
            
            return is_high_volatility
            
        except Exception as e:
            logger.error(f"Error detecting volatility for {symbol}: {str(e)}")
            return False
    
    def get_optimal_interval(self, symbol: str, data_granularity: str = '5m', smart_granularity: bool = True) -> str:
        """根据配置和市场条件确定最优的数据间隔"""
        # 如果禁用智能粒度，直接返回配置的粒度
        if not smart_granularity:
            return data_granularity
        
        # 如果配置为1分钟，直接返回
        if data_granularity == '1m':
            return '1m'
        
        # 检测高波动率
        if self.detect_high_volatility(symbol):
            logger.info(f"{symbol} high volatility detected, using 1m data")
            return '1m'
        else:
            logger.debug(f"{symbol} normal volatility, using {data_granularity} data")
            return data_granularity
    
    def aggregate_1min_to_5min(self, df_1min: pd.DataFrame) -> pd.DataFrame:
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
    
    def aggregate_to_timeframe(self, symbol: str, timeframe_minutes: int = 15, limit: int = 1000) -> List[Dict[str, Any]]:
        """将5分钟数据聚合为指定时间框架"""
        try:
            # 确保timeframe_minutes是整数类型
            if isinstance(timeframe_minutes, str):
                timeframe_minutes = int(timeframe_minutes)
            
            # 从数据库获取5分钟数据
            data = self.db_manager.get_historical_data(symbol, 5, limit * (timeframe_minutes // 5))
            
            if not data:
                logger.warning(f"No 5min data found for {symbol}")
                return []
            
            # 按时间框架分组聚合
            aggregated = []
            points_per_period = timeframe_minutes // 5  # 每个时间框架包含的5分钟数据点数
            
            for i in range(0, len(data), points_per_period):
                period_data = data[i:i + points_per_period]
                
                if len(period_data) < points_per_period:
                    continue  # 跳过不完整的周期
                
                try:
                    # 获取周期的开始时间（最新的时间戳）
                    timestamp = period_data[0][0]  # 第一条记录的时间戳
                    
                    # 计算OHLCV
                    open_price = float(period_data[-1][1])  # 周期开始价格（最早的记录）
                    close_price = float(period_data[0][2])  # 周期结束价格（最新的记录）
                    
                    # 过滤有效的高低价数据
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
            
        except Exception as e:
            logger.error(f"Error aggregating data for {symbol}: {str(e)}")
            return []
    
    def calculate_indicators_from_5min_data(self, symbol: str, limit: int = 1000) -> Optional[Dict[str, Any]]:
        """基于5分钟数据计算技术指标，支持高波动期增强模式"""
        try:
            # 从数据库获取5分钟数据
            data = self.db_manager.get_historical_data(symbol, 5, limit)
            
            if len(data) < 50:  # 需要足够的数据点计算指标
                logger.warning(f"Insufficient data for {symbol}: {len(data)} records")
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df.sort_values('timestamp')  # 按时间升序排列
            
            # 计算基础技术指标
            df['sma50'] = ta.sma(df['close'], length=50)
            df['sma200'] = ta.sma(df['close'], length=200)
            
            # 计算ADX指标
            adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
            df['adx'] = adx_data['ADX_14']
            df['plus_di'] = adx_data['DMP_14']
            df['minus_di'] = adx_data['DMN_14']
            
            # 计算布林带
            bb_data = ta.bbands(df['close'], length=20)
            df['bb_upper'] = bb_data['BBU_20_2.0']
            df['bb_middle'] = bb_data['BBM_20_2.0']
            df['bb_lower'] = bb_data['BBL_20_2.0']
            df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # 计算ATR
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # 检测高波动率
            is_high_volatility = self.detect_high_volatility(symbol)
            volatility_ratio = 1.0
            
            # 如果检测到高波动率，计算增强指标
            use_enhanced = False
            if is_high_volatility and len(df) >= 20:
                try:
                    # 计算当前ATR相对于历史平均的比率
                    current_atr = df['atr'].iloc[-1]
                    avg_atr = df['atr'].tail(20).mean()
                    volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
                    
                    # 使用更短周期的ADX指标提高敏感度
                    adx_data_short = ta.adx(df['high'], df['low'], df['close'], length=10)
                    df['adx_enhanced'] = adx_data_short['ADX_10']
                    df['plus_di_enhanced'] = adx_data_short['DMP_10']
                    df['minus_di_enhanced'] = adx_data_short['DMN_10']
                    
                    # 计算额外的波动率指标
                    df['rsi'] = ta.rsi(df['close'], length=14)  # RSI用于超买超卖判断
                    df['stoch_k'] = ta.stoch(df['high'], df['low'], df['close'])['STOCHk_14_3_3']  # 随机指标
                    
                    # 使用增强的指标值
                    use_enhanced = True
                except Exception as e:
                    logger.warning(f"Error calculating enhanced indicators: {str(e)}")
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
            
            # 获取增强模式的额外指标
            current_rsi = df['rsi'].iloc[-1] if use_enhanced and 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1]) else None
            current_stoch = df['stoch_k'].iloc[-1] if use_enhanced and 'stoch_k' in df.columns and not pd.isna(df['stoch_k'].iloc[-1]) else None
            
            # 获取有效的布林带和ATR数据
            valid_bandwidths = df['bb_bandwidth'].dropna().tail(10).tolist()
            valid_atr_values = df['atr'].dropna().tail(10).tolist()
            
            # 构建结果
            result = {
                'timestamp': df['timestamp'].iloc[-1],
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
    
    def analyze_trend(self, symbol: str, timeframe_minutes: int = 15) -> Dict[str, Any]:
        """分析趋势方向"""
        try:
            # 确保timeframe_minutes是整数类型
            if isinstance(timeframe_minutes, str):
                timeframe_minutes = int(timeframe_minutes)
            
            # 获取聚合数据
            data = self.aggregate_to_timeframe(symbol, timeframe_minutes, 100)
            
            if len(data) < 20:
                return {
                    'direction': '未知',
                    'confidence': 0,
                    'support': 0,
                    'resistance': 0,
                    'insights': '数据不足',
                    'accuracy': 0,
                    'sentiment': '中性',
                    'volatility': '中',
                    'adx': 0,
                    'trend_strength': '无'
                }
            
            # 转换为DataFrame进行分析
            df = pd.DataFrame(data)
            df = df.sort_values('timestamp')
            
            # 计算移动平均线
            df['sma_short'] = ta.sma(df['close'], length=10)
            df['sma_long'] = ta.sma(df['close'], length=20)
            
            # 计算ADX指标
            adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
            df['adx'] = adx_data['ADX_14']
            df['plus_di'] = adx_data['DMP_14']
            df['minus_di'] = adx_data['DMN_14']
            
            # 获取最新值
            current_price = df['close'].iloc[-1]
            sma_short = df['sma_short'].iloc[-1] if not pd.isna(df['sma_short'].iloc[-1]) else current_price
            sma_long = df['sma_long'].iloc[-1] if not pd.isna(df['sma_long'].iloc[-1]) else current_price
            current_adx = df['adx'].iloc[-1] if not pd.isna(df['adx'].iloc[-1]) else 0
            current_plus_di = df['plus_di'].iloc[-1] if not pd.isna(df['plus_di'].iloc[-1]) else 0
            current_minus_di = df['minus_di'].iloc[-1] if not pd.isna(df['minus_di'].iloc[-1]) else 0
            
            # 判断趋势强度
            if current_adx > 30:
                trend_strength = '强'
            elif current_adx > 20:
                trend_strength = '中'
            else:
                trend_strength = '弱'
            
            # 趋势方向判断（结合移动平均线和DMI）
            ma_trend = ''
            if current_price > sma_short > sma_long:
                ma_trend = '上涨'
            elif current_price < sma_short < sma_long:
                ma_trend = '下跌'
            else:
                ma_trend = '震荡'
            
            # DMI方向判断
            dmi_trend = ''
            if current_plus_di > current_minus_di + 2:  # 加入缓冲区避免频繁切换
                dmi_trend = '上涨'
            elif current_minus_di > current_plus_di + 2:
                dmi_trend = '下跌'
            else:
                dmi_trend = '震荡'
            
            # 综合判断趋势方向
            if ma_trend == dmi_trend and ma_trend != '震荡':
                direction = ma_trend
                base_confidence = 70
            elif ma_trend != '震荡':
                direction = ma_trend
                base_confidence = 55
            elif dmi_trend != '震荡':
                direction = dmi_trend
                base_confidence = 50
            else:
                direction = '震荡'
                base_confidence = 30
            
            # 根据ADX调整置信度
            if current_adx > 30:
                confidence = min(95, base_confidence + 20)
            elif current_adx > 20:
                confidence = min(85, base_confidence + 10)
            else:
                confidence = max(20, base_confidence - 15)
            
            # 计算支撑阻力位
            recent_lows = df['low'].tail(20)
            recent_highs = df['high'].tail(20)
            support = recent_lows.min()
            resistance = recent_highs.max()
            
            # 生成洞察信息
            insights = f'{timeframe_minutes}分钟级别{direction}趋势'
            if trend_strength != '无':
                insights += f'({trend_strength}趋势,ADX:{current_adx:.1f})'
            
            return {
                'direction': direction,
                'confidence': confidence,
                'support': support,
                'resistance': resistance,
                'insights': insights,
                'accuracy': confidence,
                'sentiment': '贪婪' if direction == '上涨' else '恐惧' if direction == '下跌' else '中性',
                'volatility': '高' if self.detect_high_volatility(symbol) else '中',
                'adx': current_adx,
                'trend_strength': trend_strength
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend for {symbol}: {str(e)}")
            return {
                'direction': '未知',
                'confidence': 0,
                'support': 0,
                'resistance': 0,
                'insights': f'分析错误: {str(e)}',
                'accuracy': 0,
                'sentiment': '中性',
                'volatility': '中'
            }