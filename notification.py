import logging
import requests
import os
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class NotificationManager:
    """通知管理器"""
    
    def __init__(self):
        self.telegram_bot_token = os.getenv('TELEGRAM_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.telegram_enabled = bool(self.telegram_bot_token and self.telegram_chat_id)
        
        if not self.telegram_enabled:
            logger.warning("Telegram notifications disabled: missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID")
    
    def send_telegram_message(self, message: str, parse_mode: str = 'Markdown') -> bool:
        """发送Telegram消息"""
        if not self.telegram_enabled:
            logger.warning("Telegram not configured, skipping message")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Failed to send Telegram message: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            return False
    
    def send_trend_alert(self, symbol: str, trend_data: Dict[str, Any]) -> bool:
        """发送趋势警报"""
        try:
            trend_direction = trend_data.get('trend_direction', 'Unknown')
            confidence = trend_data.get('confidence', 0)
            price = trend_data.get('current_price', 0)
            change_24h = trend_data.get('price_change_24h', 0)
            
            # 构建消息
            message = f"🚨 *趋势警报* 🚨\n\n"
            message += f"**币种**: {symbol}\n"
            message += f"**趋势方向**: {trend_direction}\n"
            message += f"**置信度**: {confidence:.2f}%\n"
            message += f"**当前价格**: ${price:.6f}\n"
            message += f"**24h变化**: {change_24h:+.2f}%\n"
            message += f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            # 添加趋势图标
            if trend_direction == 'BULLISH':
                message = "📈 " + message
            elif trend_direction == 'BEARISH':
                message = "📉 " + message
            else:
                message = "➡️ " + message
            
            return self.send_telegram_message(message)
            
        except Exception as e:
            logger.error(f"Error sending trend alert for {symbol}: {str(e)}")
            return False
    
    def send_quality_warning(self, symbol: str, quality_report: Dict[str, Any]) -> bool:
        """发送数据质量警告"""
        try:
            score = quality_report.get('overall_score', 0)
            grade = quality_report.get('quality_grade', 'F')
            
            # 只在质量较低时发送警告
            if score >= 70:
                return True  # 质量良好，不需要警告
            
            message = f"⚠️ *数据质量警告* ⚠️\n\n"
            message += f"**币种**: {symbol}\n"
            message += f"**质量等级**: {grade}\n"
            message += f"**质量评分**: {score:.2f}/100\n"
            
            # 添加详细信息
            data_completeness = quality_report.get('data_completeness', {})
            if data_completeness:
                message += f"\n**数据完整性**:\n"
                for timeframe, completeness in data_completeness.items():
                    pct = completeness.get('completeness_pct', 0)
                    message += f"  • {timeframe}: {pct:.1f}%\n"
            
            anomaly_detection = quality_report.get('anomaly_detection', {})
            if anomaly_detection.get('anomalies_detected'):
                message += f"\n**异常检测**:\n"
                jump_rate = anomaly_detection.get('jump_rate_pct', 0)
                zero_volume_rate = anomaly_detection.get('zero_volume_rate_pct', 0)
                max_jump = anomaly_detection.get('max_price_jump_pct', 0)
                
                if jump_rate > 1:
                    message += f"  • 价格跳跃率: {jump_rate:.1f}%\n"
                if zero_volume_rate > 5:
                    message += f"  • 零成交量率: {zero_volume_rate:.1f}%\n"
                if max_jump > 5:
                    message += f"  • 最大价格跳跃: {max_jump:.1f}%\n"
            
            message += f"\n**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return self.send_telegram_message(message)
            
        except Exception as e:
            logger.error(f"Error sending quality warning for {symbol}: {str(e)}")
            return False
    
    def send_system_status(self, status_data: Dict[str, Any]) -> bool:
        """发送系统状态报告"""
        try:
            message = f"📊 *系统状态报告* 📊\n\n"
            
            # 基本状态信息
            uptime = status_data.get('uptime', 'Unknown')
            symbols_monitored = status_data.get('symbols_monitored', 0)
            last_update = status_data.get('last_update', 'Unknown')
            
            message += f"**运行时间**: {uptime}\n"
            message += f"**监控币种**: {symbols_monitored}\n"
            message += f"**最后更新**: {last_update}\n"
            
            # API状态
            api_status = status_data.get('api_status', {})
            if api_status:
                message += f"\n**API状态**:\n"
                for api_name, status in api_status.items():
                    status_icon = "✅" if status.get('healthy', False) else "❌"
                    message += f"  {status_icon} {api_name}: {status.get('status', 'Unknown')}\n"
            
            # 数据库状态
            db_status = status_data.get('database_status', {})
            if db_status:
                message += f"\n**数据库状态**:\n"
                connection_status = "✅" if db_status.get('connected', False) else "❌"
                message += f"  {connection_status} 连接状态: {'正常' if db_status.get('connected', False) else '异常'}\n"
                
                if 'total_records' in db_status:
                    message += f"  📊 总记录数: {db_status['total_records']:,}\n"
            
            # 错误统计
            error_stats = status_data.get('error_stats', {})
            if error_stats:
                total_errors = sum(error_stats.values())
                if total_errors > 0:
                    message += f"\n**错误统计** (最近24小时):\n"
                    for error_type, count in error_stats.items():
                        if count > 0:
                            message += f"  ⚠️ {error_type}: {count}\n"
            
            message += f"\n**报告时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return self.send_telegram_message(message)
            
        except Exception as e:
            logger.error(f"Error sending system status: {str(e)}")
            return False
    
    def send_error_alert(self, error_type: str, error_message: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """发送错误警报"""
        try:
            message = f"🚨 *系统错误警报* 🚨\n\n"
            message += f"**错误类型**: {error_type}\n"
            message += f"**错误信息**: {error_message}\n"
            
            if context:
                message += f"\n**上下文信息**:\n"
                for key, value in context.items():
                    message += f"  • {key}: {value}\n"
            
            message += f"\n**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return self.send_telegram_message(message)
            
        except Exception as e:
            logger.error(f"Error sending error alert: {str(e)}")
            return False
    
    def log_notification(self, notification_type: str, recipient: str, success: bool, message: str = ""):
        """记录通知日志"""
        status = "SUCCESS" if success else "FAILED"
        log_message = f"Notification {status}: {notification_type} to {recipient}"
        
        if message:
            log_message += f" - {message[:100]}..."
        
        if success:
            logger.info(log_message)
        else:
            logger.error(log_message)