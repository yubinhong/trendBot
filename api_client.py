import requests
import time
import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class BinanceAPIClient:
    """Binance API客户端"""
    
    def __init__(self, api_rate_limit_buffer: float = 0.8):
        self.base_url = "https://api.binance.com"
        self.api_call_count = 0
        self.api_call_reset_time = time.time()
        self.API_WEIGHT_LIMIT = int(1200 * api_rate_limit_buffer)
        
        # 智能缓存系统
        self.api_cache = {}
        self.CACHE_EXPIRY_SECONDS = 30  # 缓存30秒
        self.MAX_CACHE_SIZE = 100  # 最大缓存条目数
    
    def reset_api_counter_if_needed(self):
        """如果需要，重置API调用计数器"""
        current_time = time.time()
        if current_time - self.api_call_reset_time >= 60:  # 每分钟重置
            self.api_call_count = 0
            self.api_call_reset_time = current_time
            logger.debug("API call counter reset")
    
    def generate_cache_key(self, symbol: str, interval: str, limit: int, end_time: Optional[int] = None) -> str:
        """生成缓存键"""
        key_data = f"{symbol}_{interval}_{limit}_{end_time}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_from_cache(self, cache_key: str) -> Optional[Any]:
        """从缓存获取数据"""
        if cache_key in self.api_cache:
            cached_data, timestamp = self.api_cache[cache_key]
            if time.time() - timestamp < self.CACHE_EXPIRY_SECONDS:
                logger.debug(f"Cache hit for key: {cache_key[:8]}...")
                return cached_data
            else:
                # 缓存过期，删除
                del self.api_cache[cache_key]
                logger.debug(f"Cache expired for key: {cache_key[:8]}...")
        return None
    
    def set_cache(self, cache_key: str, data: Any):
        """设置缓存"""
        # 如果缓存已满，删除最旧的条目
        if len(self.api_cache) >= self.MAX_CACHE_SIZE:
            oldest_key = min(self.api_cache.keys(), key=lambda k: self.api_cache[k][1])
            del self.api_cache[oldest_key]
            logger.debug(f"Cache full, removed oldest entry: {oldest_key[:8]}...")
        
        self.api_cache[cache_key] = (data, time.time())
        logger.debug(f"Cached data for key: {cache_key[:8]}...")
    
    def cleanup_expired_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []
        
        for key, (data, timestamp) in self.api_cache.items():
            if current_time - timestamp >= self.CACHE_EXPIRY_SECONDS:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.api_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def check_api_rate_limit(self, weight: int = 1) -> bool:
        """检查API调用频率限制"""
        self.reset_api_counter_if_needed()
        
        if self.api_call_count + weight > self.API_WEIGHT_LIMIT:
            logger.warning(f"API rate limit approaching: {self.api_call_count}/{self.API_WEIGHT_LIMIT}")
            return False
        
        self.api_call_count += weight
        return True
    
    def fetch_binance_klines(self, symbol: str, interval: Optional[str] = None, limit: int = 1000, max_retries: int = 3) -> Optional[List[Dict]]:
        """获取币安K线数据，支持智能缓存和频率控制"""
        # 使用默认间隔或传入的间隔
        if interval is None:
            interval = "5m"  # 默认5分钟
        
        # 生成缓存键并检查缓存
        cache_key = self.generate_cache_key(symbol, interval, limit)
        cached_result = self.get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        # 定期清理过期缓存
        if len(self.api_cache) > 50:  # 当缓存条目较多时清理
            self.cleanup_expired_cache()
        
        url = f"{self.base_url}/api/v3/klines"
        params = {
            'symbol': symbol.replace('/', ''),
            'interval': interval,
            'limit': limit
        }
        
        for attempt in range(max_retries):
            try:
                # 检查API频率限制
                if not self.check_api_rate_limit():
                    logger.warning("API rate limit reached, waiting...")
                    time.sleep(60)  # 等待1分钟
                    continue
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    klines_data = response.json()
                    ohlcv_data = self.convert_klines_to_ohlcv(klines_data)
                    
                    # 缓存结果
                    self.set_cache(cache_key, ohlcv_data)
                    
                    logger.debug(f"Successfully fetched {len(ohlcv_data)} klines for {symbol} ({interval})")
                    return ohlcv_data
                else:
                    logger.warning(f"API request failed with status {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # 指数退避
            
            except Exception as e:
                logger.error(f"Error fetching klines (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        logger.error(f"Failed to fetch klines for {symbol} after {max_retries} attempts")
        return None
    
    def fetch_binance_klines_with_endtime(self, symbol: str, interval: Optional[str] = None, limit: int = 1000, end_time: Optional[int] = None, max_retries: int = 3) -> Optional[List[Dict]]:
        """获取币安K线数据（带结束时间），支持智能缓存"""
        # 使用默认间隔或传入的间隔
        if interval is None:
            interval = "5m"  # 默认5分钟
        
        # 生成缓存键并检查缓存
        cache_key = self.generate_cache_key(symbol, interval, limit, end_time)
        cached_result = self.get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        url = f"{self.base_url}/api/v3/klines"
        params = {
            'symbol': symbol.replace('/', ''),
            'interval': interval,
            'limit': limit
        }
        
        if end_time:
            params['endTime'] = end_time
        
        for attempt in range(max_retries):
            try:
                # 检查API频率限制
                if not self.check_api_rate_limit():
                    logger.warning("API rate limit reached, waiting...")
                    time.sleep(60)  # 等待1分钟
                    continue
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    klines_data = response.json()
                    ohlcv_data = self.convert_klines_to_ohlcv(klines_data)
                    
                    # 缓存结果
                    self.set_cache(cache_key, ohlcv_data)
                    
                    logger.debug(f"Successfully fetched {len(ohlcv_data)} klines for {symbol} ({interval}) with endTime")
                    return ohlcv_data
                else:
                    logger.warning(f"API request failed with status {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # 指数退避
            
            except Exception as e:
                logger.error(f"Error fetching klines with endTime (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        logger.error(f"Failed to fetch klines for {symbol} after {max_retries} attempts")
        return None
    
    def convert_klines_to_ohlcv(self, klines: List[List]) -> List[Dict[str, Any]]:
        """将币安K线数据转换为OHLCV格式"""
        ohlcv_data = []
        
        for kline in klines:
            try:
                # 币安K线数据格式：[开盘时间, 开盘价, 最高价, 最低价, 收盘价, 成交量, 收盘时间, ...]
                timestamp = datetime.fromtimestamp(kline[0] / 1000)  # 转换毫秒时间戳
                
                ohlcv_data.append({
                    'timestamp': timestamp,
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing kline data: {str(e)}")
                continue
        
        # 按时间戳排序（最新的在前）
        ohlcv_data.sort(key=lambda x: x['timestamp'], reverse=True)
        
        logger.debug(f"Converted {len(ohlcv_data)} klines to OHLCV format")
        return ohlcv_data
    
    def get_api_stats(self) -> Dict[str, Any]:
        """获取API使用统计"""
        return {
            'api_call_count': self.api_call_count,
            'api_weight_limit': self.API_WEIGHT_LIMIT,
            'cache_size': len(self.api_cache),
            'cache_max_size': self.MAX_CACHE_SIZE
        }