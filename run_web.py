#!/usr/bin/env python3
"""
简单的Web界面启动脚本
用于本地开发和测试
"""

import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 检查必要的环境变量
required_vars = ['MYSQL_PASSWORD', 'TELEGRAM_BOT_TOKEN']
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

# 导入并运行Web应用
try:
    from web_app import app
    print("🚀 启动加密货币趋势监控Web界面...")
    print("📊 访问地址: http://localhost:5000")
    print("⏹️  按 Ctrl+C 停止服务")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装所有依赖: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ 启动失败: {e}")
    sys.exit(1)