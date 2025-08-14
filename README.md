# 加密货币趋势监控机器人

基于技术指标分析的加密货币趋势监控机器人，支持 BTC 和 ETH 的实时趋势分析和 Telegram 通知。

## 功能特性

- **多时间框架分析**：15分钟、1小时、4小时、1天、1周趋势监控
- **高频数据收集**：每5分钟收集5分钟K线数据并存储到数据库
- **智能分析**：基于 ADX、DMI、SMA、布林带、ATR 等技术指标
- **趋势通知**：任何时间框架趋势变化时自动发送 Telegram 通知
- **数据管理**：原始数据和趋势分析结果分别存储，自动清理90天前数据
- **Docker 部署**：完整的容器化解决方案
- **速率限制处理**：智能重试机制和API调用优化

## 快速开始

### 1. 克隆项目
```bash
git clone <your-repo-url>
cd trendBot
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API 密钥和配置
```

### 3. 启动服务
```bash
docker-compose up -d
```

### 4. 查看日志
```bash
# 查看实时日志
docker-compose logs -f trend-bot

# 查看最近的日志
docker-compose logs --tail=50 trend-bot

# 查看所有服务日志
docker-compose logs -f
```

## 环境变量说明

- `TAAPI_KEY`: TAAPI.io 的 API 密钥
- `TELEGRAM_TOKEN`: Telegram 机器人 Token
- `TELEGRAM_CHAT_ID`: Telegram 聊天 ID
- `MYSQL_HOST`: MySQL 主机地址（Docker 环境下为 `mysql`）
- `MYSQL_USER`: MySQL 用户名
- `MYSQL_PASSWORD`: MySQL 密码
- `MYSQL_DB`: MySQL 数据库名
- `MYSQL_ROOT_PASSWORD`: MySQL root 密码

## 多时间框架分析

### 时间框架
- **15分钟**: 超短期趋势，适合快速交易
- **1小时**: 短期趋势，适合日内交易
- **4小时**: 中短期趋势，适合短线操作
- **1天**: 中长期趋势，适合趋势跟踪
- **1周**: 长期趋势，适合长线投资

### 技术指标
- **ADX**: 平均趋向指数，判断趋势强度
- **+DI/-DI**: 正负方向指标，判断趋势方向
- **SMA50/200**: 50期和200期简单移动平均线
- **布林带**: 判断价格波动率
- **ATR**: 真实波动幅度

### 趋势判断逻辑
不同时间框架使用不同的阈值：
- **强趋势**: ADX > 阈值 且方向指标明确 且均线排列
- **中等趋势**: ADX > 较低阈值 且方向偏向明确
- **区间震荡**: ADX 较低 且波动率低
- **基于均线**: SMA50 与 SMA200 差异显著

### 数据存储结构
- **crypto_5min_data**: 存储5分钟原始数据和指标
- **crypto_trends**: 存储各时间框架的趋势分析结果

### 数据管理
- **数据收集频率**: 每5分钟收集一次5分钟K线数据
- **分析频率**: 每5分钟分析所有时间框架趋势
- **数据保留**: 自动清理90天前的历史数据
- **存储优化**: 使用 ON DUPLICATE KEY UPDATE 避免重复数据

## 管理命令

```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 查看日志
docker-compose logs -f trend-bot

# 进入容器
docker-compose exec trend-bot bash

# 检查容器状态
docker-compose ps

# 调试模式启动（查看详细日志）
docker-compose up --no-daemon
```