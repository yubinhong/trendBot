// 全局变量
let currentSymbol = 'BTC/USDT';

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    loadDataStatus();
    loadLatestTrends();
    loadTrendHistory();
    
    // 设置自动刷新
    setInterval(function() {
        loadLatestTrends();
    }, 60000); // 每分钟刷新一次最新趋势
});

// 加载数据状态
function loadDataStatus() {
    fetch('/api/data_status')
        .then(response => response.json())
        .then(data => {
            displayDataStatus(data);
        })
        .catch(error => {
            console.error('Error loading data status:', error);
            document.getElementById('dataStatus').innerHTML = 
                '<div class="alert alert-danger">加载数据状态失败</div>';
        });
}

// 显示数据状态
function displayDataStatus(data) {
    const container = document.getElementById('dataStatus');
    let html = '';
    
    data.forEach(status => {
        const coinName = status.symbol.split('/')[0];
        html += `
            <div class="col-md-6 mb-3">
                <div class="data-status-card">
                    <h6>${coinName}</h6>
                    <div class="display-6">${status.days_of_data} 天</div>
                    <small>${status.total_records.toLocaleString()} 条数据</small>
                    <div class="mt-2">
                        <small>最新: ${formatDateTime(status.latest_data)}</small>
                    </div>
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// 加载最新趋势
function loadLatestTrends() {
    fetch('/api/latest_trends')
        .then(response => response.json())
        .then(data => {
            displayLatestTrends(data);
        })
        .catch(error => {
            console.error('Error loading latest trends:', error);
            document.getElementById('latestTrends').innerHTML = 
                '<div class="alert alert-danger">加载最新趋势失败</div>';
        });
}

// 显示最新趋势
function displayLatestTrends(data) {
    const container = document.getElementById('latestTrends');
    
    // 按币种分组
    const groupedData = {};
    data.forEach(item => {
        if (!groupedData[item.symbol]) {
            groupedData[item.symbol] = [];
        }
        groupedData[item.symbol].push(item);
    });
    
    let html = '';
    
    Object.keys(groupedData).forEach(symbol => {
        const coinName = symbol.split('/')[0];
        const trends = groupedData[symbol];
        
        html += `
            <div class="row mb-3">
                <div class="col-12">
                    <h6><i class="fab fa-bitcoin me-2"></i>${coinName}</h6>
                    <div class="row">
        `;
        
        trends.forEach(trend => {
            const trendClass = getTrendClass(trend.trend);
            const timeframeName = getTimeframeName(trend.timeframe);
            const adxInfo = getADXInfo(trend.adx_strength);
            
            html += `
                <div class="col-md-3 col-sm-6 mb-2">
                    <div class="card border-0 bg-light">
                        <div class="card-body p-2 text-center">
                            <div class="timeframe-badge badge bg-secondary mb-1">${timeframeName}</div>
                            <div class="${trendClass}">${getTrendEmoji(trend.trend)} ${trend.trend}</div>
                            <div class="mt-1">
                                <small class="text-muted">ADX: ${trend.adx_strength ? trend.adx_strength.toFixed(1) : 'N/A'}</small>
                                <div class="adx-indicator mt-1">
                                    <div class="adx-bar ${adxInfo.class}" style="width: ${adxInfo.width}%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += `
                    </div>
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// 加载趋势历史
function loadTrendHistory() {
    const symbol = document.getElementById('symbolSelect').value;
    currentSymbol = symbol;
    
    const container = document.getElementById('trendHistory');
    container.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"></div></div>';
    
    fetch(`/api/trends/${encodeURIComponent(symbol)}`)
        .then(response => response.json())
        .then(data => {
            displayTrendHistory(data);
        })
        .catch(error => {
            console.error('Error loading trend history:', error);
            container.innerHTML = '<div class="alert alert-danger">加载历史趋势失败</div>';
        });
}

// 显示趋势历史
function displayTrendHistory(data) {
    const container = document.getElementById('trendHistory');
    
    if (data.length === 0) {
        container.innerHTML = '<div class="alert alert-info">暂无历史数据</div>';
        return;
    }
    
    let html = `
        <div class="table-responsive">
            <table class="table table-striped trend-table">
                <thead>
                    <tr>
                        <th>时间</th>
                        <th>时间框架</th>
                        <th>趋势</th>
                        <th>ADX强度</th>
                        <th>有效期(小时)</th>
                        <th>状态</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    data.forEach(item => {
        const trendClass = getTrendClass(item.trend);
        const timeframeName = getTimeframeName(item.timeframe);
        const statusBadge = item.is_expired ? 
            '<span class="badge bg-secondary">已过期</span>' : 
            '<span class="badge bg-success">有效</span>';
        
        html += `
            <tr>
                <td>${formatDateTime(item.timestamp)}</td>
                <td><span class="timeframe-badge badge bg-secondary">${timeframeName}</span></td>
                <td class="${trendClass}">${getTrendEmoji(item.trend)} ${item.trend}</td>
                <td>${item.adx_strength ? item.adx_strength.toFixed(1) : 'N/A'}</td>
                <td>${item.expected_validity_hours || 'N/A'}</td>
                <td>${statusBadge}</td>
            </tr>
        `;
    });
    
    html += `
                </tbody>
            </table>
        </div>
    `;
    
    container.innerHTML = html;
}

// 刷新最新趋势
function refreshLatestTrends() {
    const button = event.target.closest('button');
    const icon = button.querySelector('i');
    
    // 添加旋转动画
    icon.classList.add('fa-spin');
    button.disabled = true;
    
    loadLatestTrends();
    
    // 移除动画
    setTimeout(() => {
        icon.classList.remove('fa-spin');
        button.disabled = false;
    }, 1000);
}

// 工具函数
function getTrendClass(trend) {
    switch(trend) {
        case 'UP': return 'trend-up';
        case 'DOWN': return 'trend-down';
        case 'SIDEWAYS': return 'trend-sideways';
        case 'INSUFFICIENT_DATA': return 'trend-insufficient';
        default: return 'trend-unknown';
    }
}

function getTrendEmoji(trend) {
    switch(trend) {
        case 'UP': return '📈';
        case 'DOWN': return '📉';
        case 'SIDEWAYS': return '➡️';
        case 'INSUFFICIENT_DATA': return '⏳';
        default: return '❓';
    }
}

function getTimeframeName(timeframe) {
    switch(timeframe) {
        case '15m': return '15分钟';
        case '1h': return '1小时';
        case '4h': return '4小时';
        case '1d': return '1天';
        default: return timeframe;
    }
}

function getADXInfo(adxValue) {
    if (!adxValue) return { class: 'adx-weak', width: 0 };
    
    if (adxValue < 25) {
        return { class: 'adx-weak', width: (adxValue / 25) * 100 };
    } else if (adxValue < 50) {
        return { class: 'adx-moderate', width: ((adxValue - 25) / 25) * 100 };
    } else {
        return { class: 'adx-strong', width: Math.min(((adxValue - 50) / 50) * 100, 100) };
    }
}

function formatDateTime(dateString) {
    if (!dateString) return 'N/A';
    
    const date = new Date(dateString);
    return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}