// 全局变量
let currentSymbol = 'BTC/USDT';

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    loadDataStatus();
    loadLatestTrends();
    // 暂时注释掉，因为页面上没有symbolSelect元素
    // loadTrendHistory();
    
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
    
    // 检查data是否为数组，如果不是，将对象转换为数组
    const dataArray = Array.isArray(data) ? data : Object.entries(data).map(([symbol, info]) => {
        return {
            symbol: symbol,
            days_available: info.days_available || 0,
            total_records: info.total_records || 0,
            latest_data: info.newest_record || new Date().toISOString()
        };
    });
    
    if (dataArray.length === 0) {
        html = '<div class="alert alert-warning">暂无数据</div>';
    } else {
        html = '<div class="row">';
        dataArray.forEach(status => {
            const coinName = status.symbol.split('/')[0];
            html += `
                <div class="col-md-6 mb-3">
                    <div class="data-status-card">
                        <h6>${coinName}</h6>
                        <div class="display-6">${status.days_available || 0} 天</div>
                        <small>${(status.total_records || 0).toLocaleString()} 条数据</small>
                        <div class="mt-2">
                            <small>最新: ${formatDateTime(status.latest_data)}</small>
                        </div>
                    </div>
                </div>
            `;
        });
        html += '</div>';
    }
    
    
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
            document.getElementById('trendsContainer').innerHTML = 
                '<div class="alert alert-danger">加载最新趋势失败</div>';
        });
}

// 显示最新趋势
function displayLatestTrends(data) {
    const container = document.getElementById('trendsContainer');
    
    // 检查data是否为数组，如果不是，使用原始数据结构
    if (!Array.isArray(data)) {
        // 数据已经按币种分组
        displayGroupedTrends(data, container);
        return;
    }
    
    // 如果是数组，按币种分组
    const groupedData = {};
    data.forEach(item => {
        if (!groupedData[item.symbol]) {
            groupedData[item.symbol] = [];
        }
        groupedData[item.symbol].push(item);
    });
    
    // 使用分组后的数据显示趋势
    displayGroupedTrends(groupedData, container);
}

// 添加displayGroupedTrends函数
function displayGroupedTrends(groupedData, container) {
    let html = '';
    
    if (!groupedData || Object.keys(groupedData).length === 0) {
        container.innerHTML = '<div class="alert alert-info">暂无趋势数据</div>';
        return;
    }
    
    Object.keys(groupedData).forEach(symbol => {
        const coinName = symbol.split('/')[0];
        const trends = groupedData[symbol];
        
        html += `
            <div class="card mb-4">
                <div class="card-header bg-dark text-white">
                    <h5 class="mb-0">${coinName} 趋势</h5>
                </div>
                <div class="card-body">
                    <div class="row">
        `;
        
        // 确保trends是数组
        const trendsArray = Array.isArray(trends) ? trends : [trends];
        
        trendsArray.forEach(trend => {
            // 确保trend对象有所需的属性
            const trendDirection = trend.trend || 'UNKNOWN';
            const timeframe = trend.timeframe || 'UNKNOWN';
            const adxStrength = trend.adx_strength || 0;
            
            const trendClass = getTrendClass(trendDirection);
            const timeframeName = getTimeframeName(timeframe);
            const adxInfo = getADXInfo(adxStrength);
            
            html += `
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="card trend-card ${trendClass}">
                        <div class="card-body p-3 text-center">
                            <div class="timeframe-badge badge bg-secondary mb-2">${timeframeName}</div>
                            <div class="fs-5">${getTrendEmoji(trendDirection)} ${trendDirection}</div>
                            <div class="mt-2">
                                <small class="text-muted">ADX: ${adxStrength ? adxStrength.toFixed(1) : 'N/A'}</small>
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
function loadTrendHistory(symbol) {
    if (!symbol) {
        symbol = currentSymbol; // 使用全局变量中的默认值
    }
    
    // 检查是否存在trendHistory元素
    const container = document.getElementById('trendHistory');
    if (!container) {
        console.warn('trendHistory容器不存在');
        return;
    }
    
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