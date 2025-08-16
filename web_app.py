#!/usr/bin/env python3
"""
加密货币趋势监控Web界面
提供数据可视化和趋势分析
"""

import os
import sys
import json
import mysql.connector
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, send_from_directory
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 检查必要的环境变量
required_vars = ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DB']
missing_vars = []

for var in required_vars:
    if not os.getenv(var):
        missing_vars.append(var)

if missing_vars:
    print("❌ 缺少必要的环境变量:")
    for var in missing_vars:
        print(f"   - {var}")
    print("\n请检查 .env 文件或设置环境变量")
    sys.exit(1)

# 数据库连接信息
MYSQL_HOST = os.getenv('MYSQL_HOST', 'mysql')
MYSQL_USER = os.getenv('MYSQL_USER', 'user')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_DB = os.getenv('MYSQL_DB', 'crypto_trends')

# 创建Flask应用
app = Flask(__name__, static_folder='static')

# 数据库连接函数
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        return conn
    except Exception as e:
        print(f"❌ 数据库连接错误: {str(e)}")
        return None

# 主页路由
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# 静态文件路由
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# API路由 - 获取所有趋势数据
@app.route('/api/trends')
@app.route('/api/latest_trends')  # 添加兼容app.js的路由
def get_trends():
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '数据库连接失败'}), 500
        
        cursor = conn.cursor(dictionary=True)
        
        # 获取最近的趋势数据
        cursor.execute("""
            SELECT * FROM crypto_trends 
            WHERE is_expired = FALSE 
            ORDER BY symbol, timeframe, timestamp DESC
        """)
        
        all_trends = cursor.fetchall()
        
        # 处理日期格式
        for trend in all_trends:
            if 'timestamp' in trend and trend['timestamp']:
                trend['timestamp'] = trend['timestamp'].isoformat()
        
        # 按币种和时间框架分组
        grouped_trends = {}
        for trend in all_trends:
            symbol = trend['symbol']
            timeframe = trend['timeframe']
            
            if symbol not in grouped_trends:
                grouped_trends[symbol] = []
            
            # 将趋势添加到对应币种的列表中
            grouped_trends[symbol].append(trend)
        
        cursor.close()
        conn.close()
        
        # 返回按币种分组的趋势数据，这种格式与app.js中的displayGroupedTrends函数兼容
        return jsonify(grouped_trends)
    
    except Exception as e:
        return jsonify({'error': f'获取趋势数据失败: {str(e)}'}), 500

# API路由 - 获取数据状态
@app.route('/api/data-status')
@app.route('/api/data_status')  # 添加兼容app.js的路由
def get_data_status():
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '数据库连接失败'}), 500
        
        cursor = conn.cursor(dictionary=True)
        
        # 获取5分钟数据统计
        cursor.execute("""
            SELECT symbol, COUNT(*) as total_records, 
                   MIN(timestamp) as oldest_record, 
                   MAX(timestamp) as newest_record
            FROM crypto_5min_data 
            GROUP BY symbol
        """)
        
        data_stats = {}
        for row in cursor.fetchall():
            symbol = row['symbol']
            data_stats[symbol] = {
                'total_records': row['total_records'],
                'oldest_record': row['oldest_record'].isoformat() if row['oldest_record'] else None,
                'newest_record': row['newest_record'].isoformat() if row['newest_record'] else None,
                'days_available': (row['newest_record'] - row['oldest_record']).days if row['newest_record'] and row['oldest_record'] else 0
            }
        
        cursor.close()
        conn.close()
        
        return jsonify(data_stats)
    
    except Exception as e:
        return jsonify({'error': f'获取数据状态失败: {str(e)}'}), 500

# API路由 - 获取特定币种的趋势数据
@app.route('/api/trends/<symbol>')
def get_symbol_trends(symbol):
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '数据库连接失败'}), 500
        
        cursor = conn.cursor(dictionary=True)
        
        # 获取指定币种的趋势数据
        cursor.execute("""
            SELECT * FROM crypto_trends 
            WHERE symbol = %s
            ORDER BY timestamp DESC
            LIMIT 50
        """, (symbol,))
        
        trends = cursor.fetchall()
        
        # 处理日期格式
        for trend in trends:
            if 'timestamp' in trend and trend['timestamp']:
                trend['timestamp'] = trend['timestamp'].isoformat()
        
        cursor.close()
        conn.close()
        
        return jsonify(trends)
    
    except Exception as e:
        return jsonify({'error': f'获取币种趋势数据失败: {str(e)}'}), 500

# API路由 - 获取历史趋势数据
@app.route('/api/historical-trends')
def get_historical_trends():
    try:
        symbol = request.args.get('symbol')
        timeframe = request.args.get('timeframe')
        days = int(request.args.get('days', 7))
        
        if not symbol or not timeframe:
            return jsonify({'error': '缺少必要参数'}), 400
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '数据库连接失败'}), 500
        
        cursor = conn.cursor(dictionary=True)
        
        # 获取指定时间范围内的趋势数据
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT * FROM crypto_trends 
            WHERE symbol = %s AND timeframe = %s AND timestamp >= %s
            ORDER BY timestamp ASC
        """, (symbol, timeframe, start_date))
        
        trends = cursor.fetchall()
        
        # 处理日期格式
        for trend in trends:
            if 'timestamp' in trend and trend['timestamp']:
                trend['timestamp'] = trend['timestamp'].isoformat()
        
        cursor.close()
        conn.close()
        
        return jsonify(trends)
    
    except Exception as e:
        return jsonify({'error': f'获取历史趋势数据失败: {str(e)}'}), 500

# 启动应用
if __name__ == "__main__":
    print("🚀 启动加密货币趋势监控Web界面...")
    print("📊 访问地址: http://0.0.0.0:5000")
    print("⏹️  按 Ctrl+C 停止服务")
    
    app.run(host='0.0.0.0', port=5000, debug=False)