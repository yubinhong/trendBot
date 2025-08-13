# 加密货币趋势监控机器人

基于技术指标分析的加密货币趋势监控机器人，支持 BTC 和 ETH 的实时趋势分析和 Telegram 通知。

## 功能特性

- 实时监控 BTC/USDT 和 ETH/USDT 趋势
- 基于 ADX、DMI、SMA、布林带、ATR 等技术指标进行分析
- 趋势变化时自动发送 Telegram 通知
- 数据持久化存储到 MySQL 数据库
- Docker 容器化部署

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
docker-compose logs -f trend-bot
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

## 技术指标说明

- **ADX**: 平均趋向指数，判断趋势强度
- **+DI/-DI**: 正负方向指标，判断趋势方向
- **SMA50/200**: 50日和200日简单移动平均线
- **布林带**: 判断价格波动率
- **ATR**: 真实波动幅度

## 趋势判断逻辑

- **上涨趋势**: ADX > 25 且 +DI > -DI 且 SMA50 > SMA200
- **下跌趋势**: ADX > 25 且 -DI > +DI 且 SMA50 < SMA200
- **区间震荡**: ADX ≤ 25 且波动率较低

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
```