import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from database import DatabaseManager
from api_client import BinanceAPIClient

logger = logging.getLogger(__name__)

class DataQualityMonitor:
    """数据质量监控器"""
    
    def __init__(self, db_manager: DatabaseManager, api_client: BinanceAPIClient):
        self.db_manager = db_manager
        self.api_client = api_client
    
    def check_data_completeness(self, symbol: str, timeframe: str = '5m', hours_back: int = 24) -> Dict[str, Any]:
        """检查数据完整性"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # 计算预期的数据点数量
            if timeframe == '1m':
                expected_points = hours_back * 60
                table = 'crypto_1min_data'
            elif timeframe == '5m':
                expected_points = hours_back * 12
                table = 'crypto_5min_data'
            else:
                expected_points = hours_back
                table = 'crypto_5min_data'
            
            # 查询实际数据点数量
            cursor.execute(f"""
                SELECT COUNT(*) FROM {table}
                WHERE symbol = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
            """, (symbol, hours_back))
            
            actual_points = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            completeness_pct = (actual_points / expected_points) * 100 if expected_points > 0 else 0
            
            return {
                'timeframe': timeframe,
                'expected_points': expected_points,
                'actual_points': actual_points,
                'completeness_pct': completeness_pct,
                'is_complete': completeness_pct >= 95
            }
            
        except Exception as e:
            logger.error(f"Error checking data completeness for {symbol}: {str(e)}")
            return {
                'timeframe': timeframe,
                'expected_points': 0,
                'actual_points': 0,
                'completeness_pct': 0,
                'is_complete': False
            }
    
    def compare_aggregated_vs_direct(self, symbol: str, sample_size: int = 50) -> Dict[str, Any]:
        """比较聚合数据与直接API数据的差异"""
        try:
            # 获取最近的5分钟聚合数据
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT timestamp, open_price, high_price, low_price, close_price, volume
                FROM crypto_5min_data
                WHERE symbol = %s AND data_source = 'aggregated_1m'
                ORDER BY timestamp DESC
                LIMIT %s
            """, (symbol, sample_size))
            
            aggregated_data = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not aggregated_data:
                return {
                    'comparison_available': False,
                    'reason': 'No aggregated data found'
                }
            
            # 获取对应时间段的直接API数据
            api_data = self.api_client.fetch_binance_klines(symbol, '5m', sample_size)
            
            if not api_data:
                return {
                    'comparison_available': False,
                    'reason': 'Failed to fetch API data'
                }
            
            # 比较数据差异
            price_differences = []
            volume_differences = []
            
            for agg_record in aggregated_data[:min(len(aggregated_data), len(api_data))]:
                # 找到对应时间的API数据
                agg_timestamp = agg_record[0]
                matching_api = None
                
                for api_record in api_data:
                    if abs((api_record['timestamp'] - agg_timestamp).total_seconds()) < 300:  # 5分钟内
                        matching_api = api_record
                        break
                
                if matching_api:
                    # 计算价格差异
                    price_diff = abs(agg_record[4] - matching_api['close']) / matching_api['close'] * 100
                    volume_diff = abs(agg_record[5] - matching_api['volume']) / matching_api['volume'] * 100 if matching_api['volume'] > 0 else 0
                    
                    price_differences.append(price_diff)
                    volume_differences.append(volume_diff)
            
            if price_differences:
                avg_price_diff = np.mean(price_differences)
                max_price_diff = np.max(price_differences)
                avg_volume_diff = np.mean(volume_differences)
                
                return {
                    'comparison_available': True,
                    'samples_compared': len(price_differences),
                    'avg_price_difference_pct': avg_price_diff,
                    'max_price_difference_pct': max_price_diff,
                    'avg_volume_difference_pct': avg_volume_diff,
                    'data_quality_score': max(0, 100 - avg_price_diff * 10)  # 简单评分
                }
            else:
                return {
                    'comparison_available': False,
                    'reason': 'No matching timestamps found'
                }
                
        except Exception as e:
            logger.error(f"Error comparing aggregated vs direct data for {symbol}: {str(e)}")
            return {
                'comparison_available': False,
                'reason': f'Error: {str(e)}'
            }
    
    def detect_price_anomalies(self, symbol: str, timeframe: str = '5m', hours_back: int = 24) -> Dict[str, Any]:
        """检测价格异常和数据跳跃"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            table = 'crypto_1min_data' if timeframe == '1m' else 'crypto_5min_data'
            
            cursor.execute(f"""
                SELECT timestamp, close_price, volume
                FROM {table}
                WHERE symbol = %s AND timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
                ORDER BY timestamp ASC
            """, (symbol, hours_back))
            
            data = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if len(data) < 10:
                return {
                    'anomalies_detected': False,
                    'reason': 'Insufficient data for anomaly detection'
                }
            
            # 转换为DataFrame进行分析
            df = pd.DataFrame(data, columns=['timestamp', 'price', 'volume'])
            
            # 计算价格变化率
            df['price_change_pct'] = df['price'].pct_change() * 100
            
            # 检测异常价格跳跃（超过5%的变化）
            large_jumps = df[abs(df['price_change_pct']) > 5]
            
            # 检测零成交量
            zero_volume = df[df['volume'] == 0]
            
            # 检测重复价格（可能的数据停滞）
            df['price_diff'] = df['price'].diff()
            stagnant_periods = df[df['price_diff'] == 0]
            
            # 计算统计信息
            jump_rate = len(large_jumps) / len(df) * 100
            zero_volume_rate = len(zero_volume) / len(df) * 100
            stagnant_rate = len(stagnant_periods) / len(df) * 100
            
            # 检测高波动期
            high_volatility_periods = df[abs(df['price_change_pct']) > 2]
            
            return {
                'anomalies_detected': True,
                'total_records': len(df),
                'large_jumps': len(large_jumps),
                'jump_rate_pct': jump_rate,
                'zero_volume_periods': len(zero_volume),
                'zero_volume_rate_pct': zero_volume_rate,
                'stagnant_periods': len(stagnant_periods),
                'stagnant_rate_pct': stagnant_rate,
                'high_volatility_periods': len(high_volatility_periods),
                'max_price_jump_pct': abs(df['price_change_pct']).max() if not df['price_change_pct'].isna().all() else 0
            }
            
        except Exception as e:
            logger.error(f"Error detecting price anomalies for {symbol}: {str(e)}")
            return {
                'anomalies_detected': False,
                'reason': f'Error: {str(e)}'
            }
    
    def calculate_quality_score(self, symbol: str) -> Dict[str, Any]:
        """计算整体数据质量评分"""
        try:
            # 检查各个时间框架的数据完整性
            completeness_1m = self.check_data_completeness(symbol, '1m', 24)
            completeness_5m = self.check_data_completeness(symbol, '5m', 24)
            
            # 检查数据异常
            anomalies_5m = self.detect_price_anomalies(symbol, '5m', 24)
            
            # 比较聚合数据质量
            comparison = self.compare_aggregated_vs_direct(symbol)
            
            # 计算综合评分
            score = 100
            
            # 数据完整性评分（40%权重）
            completeness_score = (completeness_1m['completeness_pct'] + completeness_5m['completeness_pct']) / 2
            score = score * 0.6 + completeness_score * 0.4
            
            # 异常检测评分（30%权重）
            if anomalies_5m['anomalies_detected']:
                anomaly_penalty = min(30, anomalies_5m['jump_rate_pct'] * 5 + anomalies_5m['zero_volume_rate_pct'] * 3)
                score -= anomaly_penalty
            
            # 数据一致性评分（30%权重）
            if comparison['comparison_available']:
                consistency_score = comparison.get('data_quality_score', 80)
                score = score * 0.7 + consistency_score * 0.3
            
            # 确保评分在0-100范围内
            score = max(0, min(100, score))
            
            # 确定质量等级
            if score >= 90:
                quality_grade = 'A'
            elif score >= 80:
                quality_grade = 'B'
            elif score >= 70:
                quality_grade = 'C'
            elif score >= 60:
                quality_grade = 'D'
            else:
                quality_grade = 'F'
            
            return {
                'symbol': symbol,
                'overall_score': round(score, 2),
                'quality_grade': quality_grade,
                'timestamp': datetime.now(),
                'data_completeness': {
                    '1m': completeness_1m,
                    '5m': completeness_5m
                },
                'anomaly_detection': anomalies_5m,
                'data_comparison': comparison
            }
            
        except Exception as e:
            logger.error(f"Error calculating quality score for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'overall_score': 0,
                'quality_grade': 'F',
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    def monitor_data_quality(self, symbols: List[str]) -> Dict[str, Any]:
        """监控多个币种的数据质量"""
        try:
            quality_reports = {}
            overall_stats = {
                'total_symbols': len(symbols),
                'high_quality_count': 0,
                'medium_quality_count': 0,
                'low_quality_count': 0,
                'average_score': 0
            }
            
            total_score = 0
            
            for symbol in symbols:
                logger.info(f"Monitoring data quality for {symbol}")
                quality_report = self.calculate_quality_score(symbol)
                quality_reports[symbol] = quality_report
                
                score = quality_report['overall_score']
                total_score += score
                
                # 统计质量分布
                if score >= 80:
                    overall_stats['high_quality_count'] += 1
                elif score >= 60:
                    overall_stats['medium_quality_count'] += 1
                else:
                    overall_stats['low_quality_count'] += 1
            
            overall_stats['average_score'] = total_score / len(symbols) if symbols else 0
            
            logger.info(f"Data quality monitoring completed. Average score: {overall_stats['average_score']:.2f}")
            
            return {
                'monitoring_timestamp': datetime.now(),
                'quality_reports': quality_reports,
                'overall_stats': overall_stats
            }
            
        except Exception as e:
            logger.error(f"Error monitoring data quality: {str(e)}")
            return {
                'monitoring_timestamp': datetime.now(),
                'error': str(e),
                'quality_reports': {},
                'overall_stats': {
                    'total_symbols': 0,
                    'high_quality_count': 0,
                    'medium_quality_count': 0,
                    'low_quality_count': 0,
                    'average_score': 0
                }
            }