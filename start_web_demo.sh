#!/bin/bash

echo "🚀 启动加密货币趋势监控Web界面演示"
echo "=================================="

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker未运行，请先启动Docker"
    exit 1
fi

# 检查.env文件
if [ ! -f .env ]; then
    echo "⚠️  未找到.env文件，从示例文件创建..."
    cp .env.example .env
    echo "📝 请编辑.env文件，填入必要的配置信息"
    echo "   - MYSQL_ROOT_PASSWORD"
    echo "   - MYSQL_PASSWORD"
    echo "   - TELEGRAM_BOT_TOKEN"
    echo "   - TELEGRAM_CHAT_ID"
    echo ""
    echo "配置完成后重新运行此脚本"
    exit 1
fi

echo "📦 启动服务..."
docker-compose up -d

echo ""
echo "⏳ 等待服务启动..."
sleep 10

echo ""
echo "✅ 服务启动完成！"
echo ""
echo "📊 Web界面地址: http://localhost:5000"
echo "📱 移动端友好，支持手机访问"
echo ""
echo "🔍 查看服务状态:"
echo "   docker-compose ps"
echo ""
echo "📋 查看日志:"
echo "   docker-compose logs -f trend-bot      # 趋势机器人日志"
echo "   docker-compose logs -f web-dashboard  # Web界面日志"
echo ""
echo "⏹️  停止服务:"
echo "   docker-compose down"
echo ""
echo "🎯 首次运行需要一些时间收集数据，请耐心等待..."