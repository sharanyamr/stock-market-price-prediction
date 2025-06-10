// Portfolio management functionality

document.addEventListener('DOMContentLoaded', function() {
    console.log('Portfolio.js initialized');
    
    // Fetch portfolio data and initialize all components
    fetchPortfolioData();
    
    // Fetch recent transactions
    fetchRecentTransactions();
    
    // Setup add stock form validation
    setupAddStockForm();
    
    // Setup sell buttons
    setupSellButtons();
});

// Cache for portfolio data
let portfolioDataCache = null;
let lastFetchTime = 0;
const CACHE_DURATION = 60000; // 1 minute cache in milliseconds

// Function to show empty portfolio message or error
function showEmptyPortfolioMessage(message = 'No stocks in your portfolio') {
    console.log('Showing empty portfolio message:', message);
    
    // Always remove loading state from portfolio table
    const portfolioTable = document.getElementById('portfolioTable');
    if (portfolioTable) {
        portfolioTable.classList.remove('loading');
    }
    
    // Find or create table body
    let tableBody = document.getElementById('portfolioTableBody');
    if (!tableBody && portfolioTable) {
        tableBody = document.createElement('tbody');
        tableBody.id = 'portfolioTableBody';
        portfolioTable.appendChild(tableBody);
    }
    
    if (tableBody) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="7" class="text-center py-4">
                    <i class="fas fa-info-circle text-muted fa-2x mb-3"></i>
                    <p>${message}</p>
                    <a href="/portfolio/add" class="btn btn-sm btn-primary mt-2">
                        <i class="fas fa-plus-circle me-1"></i> Add Stock
                    </a>
                </td>
            </tr>
        `;
    }
    
    // Make sure portfolio holdings card is visible
    const holdingsCard = document.getElementById('portfolioHoldingsCard');
    if (holdingsCard) {
        holdingsCard.style.display = 'block';
    }
}

// Fetch portfolio data from API with caching
function fetchPortfolioData(forceRefresh = false) {
    const now = Date.now();
    const cacheExpired = (now - lastFetchTime) > CACHE_DURATION;
    
    // Use cached data if available and not expired
    if (!forceRefresh && portfolioDataCache && !cacheExpired) {
        console.log('Using cached portfolio data');
        processPortfolioData(portfolioDataCache);
        return;
    }
    
    // Show loading indicator
    const portfolioTable = document.getElementById('portfolioTable');
    if (portfolioTable) {
        portfolioTable.classList.add('loading');
    }
    
    // Also make sure portfolio holdings card is visible
    const holdingsCard = document.getElementById('portfolioHoldingsCard');
    if (holdingsCard) {
        holdingsCard.style.display = 'block';
        console.log('Made portfolio holdings card visible');
    }
    
    console.log('Fetching fresh portfolio data...');
    
    // Use AbortController to timeout fetch requests that take too long
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
    
    fetch('/portfolio/data', {
        signal: controller.signal,
        headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
    })
        .then(response => {
            clearTimeout(timeoutId);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Portfolio data loaded:', data);
            
            // Update cache
            portfolioDataCache = data;
            lastFetchTime = now;
            
            // Process the data
            processPortfolioData(data);
        })
        .catch(error => {
            clearTimeout(timeoutId);
            console.error('Error fetching portfolio data:', error);
            
            // If we have cached data, use it as fallback
            if (portfolioDataCache) {
                console.log('Using cached data as fallback after error');
                processPortfolioData(portfolioDataCache);
            } else {
                showEmptyPortfolioMessage(error.message);
                
                // Make sure the portfolio holdings card is visible even on error
                const holdingsCard = document.getElementById('portfolioHoldingsCard');
                if (holdingsCard) {
                    holdingsCard.style.display = 'block';
                }
            }
        })
        .finally(() => {
            // Remove loading indicator
            const portfolioTable = document.getElementById('portfolioTable');
            if (portfolioTable) {
                portfolioTable.classList.remove('loading');
            }
            
            // Also update the table body to ensure it's not stuck in loading state
            const tableBody = document.getElementById('portfolioTableBody');
            if (tableBody && tableBody.querySelector('.spinner-border')) {
                // If there's still a spinner, replace it with a message
                showEmptyPortfolioMessage('Could not load portfolio data. Please try again later.');
            }
        });
}

// Process portfolio data
function processPortfolioData(data) {
    console.log('Processing portfolio data:', data);
    
    // Always remove loading state from portfolio table
    const portfolioTable = document.getElementById('portfolioTable');
    if (portfolioTable) {
        portfolioTable.classList.remove('loading');
        console.log('Removed loading class from portfolio table');
    }
    
    if (!data) {
        console.error('No data received from server');
        showEmptyPortfolioMessage('No data received from server');
        return;
    }
    
    // Make sure portfolio holdings card is visible
    const holdingsCard = document.getElementById('portfolioHoldingsCard');
    if (holdingsCard) {
        holdingsCard.style.display = 'block';
        console.log('Made portfolio holdings card visible');
    } else {
        console.warn('Could not find portfolioHoldingsCard element');
    }
    
    if (data.error) {
        console.error('Error in data:', data.error);
        showEmptyPortfolioMessage(data.error);
        return;
    }
    
    // Handle different data formats
    let portfolioItems = [];
    let totals = {};
    
    if (Array.isArray(data)) {
        // Old format: array of portfolio items
        console.log('Data is in old array format');
        portfolioItems = data;
    } else if (data.portfolio_items) {
        // New format: object with portfolio_items and totals
        console.log('Data is in new object format');
        portfolioItems = data.portfolio_items;
        totals = data.totals || {};
    } else {
        console.error('Unknown data format:', data);
        showEmptyPortfolioMessage('Unknown data format');
        return;
    }
    
    console.log('Portfolio items:', portfolioItems);
    
    if (!portfolioItems || portfolioItems.length === 0) {
        console.log('Empty portfolio data array');
        showEmptyPortfolioMessage('No stocks in your portfolio');
        return;
    }
    
    // If totals are not provided from the backend, calculate them here
    if (!totals.total_cost && !totals.total_value) {
        console.log('Calculating portfolio totals on client side');
        totals = calculatePortfolioTotals(portfolioItems);
    }
    
    // Update UI with portfolio data
    updatePortfolioSummary(portfolioItems, totals);
    updatePortfolioTable(portfolioItems);
    
    // Initialize charts
    initPortfolioSummaryChart(portfolioItems);
    initPortfolioPerformanceChart(portfolioItems);
    
    // Update stock information component with the best performing stock
    updateStockInfoComponent(portfolioItems);
    
    // Show portfolio sections if they exist (they might not on all pages)
    const analysisCard = document.getElementById('portfolioAnalysisCard');
    
    if (analysisCard) analysisCard.style.display = 'block';
}

// Calculate portfolio totals
function calculatePortfolioTotals(portfolioData) {
    let totalValue = 0;
    let totalCost = 0;
    let totalProfitLoss = 0;
    
    portfolioData.forEach(item => {
        totalValue += item.current_value;
        totalCost += item.cost_basis;
        totalProfitLoss += item.profit_loss;
    });
    
    const totalProfitLossPercent = totalCost > 0 ? (totalProfitLoss / totalCost) * 100 : 0;
    
    return {
        totalValue,
        totalCost,
        totalProfitLoss,
        totalProfitLossPercent
    };
}

// Format currency with commas for better readability
function formatCurrency(value, currency = '₹') {
    return `${currency}${value.toLocaleString('en-IN', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    })}`;
}

// Show empty portfolio message or error message
function showEmptyPortfolioMessage(message = 'No stocks in your portfolio') {
    console.log('Showing empty portfolio message:', message);
    
    // Update the portfolio table body with a message
    const tableBody = document.getElementById('portfolioTableBody');
    if (tableBody) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="7" class="text-center">
                    <i class="fas fa-info-circle text-muted fa-2x mb-3"></i>
                    <p>${message}</p>
                    <a href="/portfolio/add" class="btn btn-sm btn-primary mt-2">
                        <i class="fas fa-plus-circle me-1"></i> Add Stock
                    </a>
                </td>
            </tr>
        `;
    }
    
    // Update summary cards with zeros if they exist
    const totalValueDisplay = document.getElementById('totalValueDisplay');
    if (totalValueDisplay) {
        totalValueDisplay.innerHTML = '₹0.00';
    }
    
    const totalCostDisplay = document.getElementById('totalCostDisplay');
    if (totalCostDisplay) {
        totalCostDisplay.innerHTML = '₹0.00';
    }
    
    const profitLossDisplay = document.getElementById('profitLossDisplay');
    if (profitLossDisplay) {
        profitLossDisplay.innerHTML = '₹0.00';
    }
    
    const profitLossPercentDisplay = document.getElementById('profitLossPercentDisplay');
    if (profitLossPercentDisplay) {
        profitLossPercentDisplay.innerHTML = '0.00%';
    }
    
    const holdingsCountDisplay = document.getElementById('holdingsCountDisplay');
    if (holdingsCountDisplay) {
        holdingsCountDisplay.innerHTML = '0';
    }
}

// Update portfolio summary cards
function updatePortfolioSummary(portfolioData, totals) {
    // Update total value
    const totalValueDisplay = document.getElementById('totalValueDisplay');
    if (totals.total_value_formatted) {
        totalValueDisplay.innerHTML = totals.total_value_formatted;
    } else {
        totalValueDisplay.innerHTML = formatCurrency(totals.total_value || 0);
    }
    
    // Update total cost
    const totalCostDisplay = document.getElementById('totalCostDisplay');
    if (totals.total_cost_formatted) {
        totalCostDisplay.innerHTML = totals.total_cost_formatted;
    } else {
        totalCostDisplay.innerHTML = formatCurrency(totals.total_cost || 0);
    }
    
    // Update profit/loss
    const profitLossDisplay = document.getElementById('profitLossDisplay');
    const profitLossValue = totals.total_profit_loss || 0;
    const profitLossClass = profitLossValue >= 0 ? 'text-success' : 'text-danger';
    
    // Format profit/loss with proper sign and currency
    if (totals.total_profit_loss_formatted) {
        // Use the formatted value from the server if available
        // The server now includes the + sign for positive values
        profitLossDisplay.innerHTML = `<span class="${profitLossClass}">${totals.total_profit_loss_formatted}</span>`;
    } else {
        // Format locally if server didn't provide formatted value
        const profitLossSign = profitLossValue >= 0 ? '+' : '-';
        profitLossDisplay.innerHTML = `<span class="${profitLossClass}">${profitLossSign}${formatCurrency(Math.abs(profitLossValue))}</span>`;
    }
    
    // Log for debugging
    console.log('Profit/Loss Value:', profitLossValue);
    console.log('Profit/Loss Formatted:', totals.total_profit_loss_formatted);
    
    // Update profit/loss percentage
    const profitLossPercentDisplay = document.getElementById('profitLossPercentDisplay');
    const profitLossPercentValue = totals.total_profit_loss_percent || 0;
    const profitLossPercentClass = profitLossPercentValue >= 0 ? 'text-success' : 'text-danger';
    const profitLossPercentSign = profitLossPercentValue >= 0 ? '+' : '';
    profitLossPercentDisplay.innerHTML = `<span class="${profitLossPercentClass}">${profitLossPercentSign}${profitLossPercentValue.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}%</span>`;
    
    // Update holdings count
    const holdingsCountDisplay = document.getElementById('holdingsCountDisplay');
    holdingsCountDisplay.innerHTML = portfolioData.length;
    
    // Update analysis card
    if (document.getElementById('totalInvestmentCell')) {
        if (totals.total_cost_formatted) {
            document.getElementById('totalInvestmentCell').innerHTML = totals.total_cost_formatted;
        } else {
            document.getElementById('totalInvestmentCell').innerHTML = formatCurrency(totals.total_cost || 0);
        }
    }
    
    if (document.getElementById('currentValueCell')) {
        if (totals.total_value_formatted) {
            document.getElementById('currentValueCell').innerHTML = totals.total_value_formatted;
        } else {
            document.getElementById('currentValueCell').innerHTML = formatCurrency(totals.total_value || 0);
        }
    }
    
    const profitLossCell = document.getElementById('profitLossCell');
    if (profitLossCell) {
        profitLossCell.className = `text-end fw-bold ${profitLossValue >= 0 ? 'text-success' : 'text-danger'}`;
        
        if (totals.total_profit_loss_formatted) {
            // Use the formatted value directly from the server
            // The server now includes the proper sign
            profitLossCell.innerHTML = totals.total_profit_loss_formatted;
        } else {
            // Format locally if server didn't provide formatted value
            const sign = profitLossValue >= 0 ? '+' : '-';
            profitLossCell.innerHTML = `${sign}${formatCurrency(Math.abs(profitLossValue))}`;
        }
        
        // Log for debugging
        console.log('Profit/Loss Cell Updated:', profitLossCell.innerHTML);
    }
    
    const profitLossPercentCell = document.getElementById('profitLossPercentCell');
    if (profitLossPercentCell) {
        profitLossPercentCell.className = `text-end fw-bold ${profitLossPercentValue >= 0 ? 'text-success' : 'text-danger'}`;
        profitLossPercentCell.innerHTML = `${profitLossPercentSign}${profitLossPercentValue.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}%`;
    }
}

// Initialize portfolio summary chart
function initPortfolioSummaryChart(portfolioData) {
    const summaryChartCanvas = document.getElementById('portfolioSummaryChart');
    if (!summaryChartCanvas) return;
    
    // Prepare data for pie chart
    const composition = {};
    let totalValue = 0;
    
    // Calculate total value and composition
    portfolioData.forEach(item => {
        totalValue += item.current_value;
    });
    
    // Calculate percentages
    portfolioData.forEach(item => {
        composition[item.symbol] = (item.current_value / totalValue) * 100;
    });
    
    const labels = Object.keys(composition);
    const values = Object.values(composition);
    
    // Generate colors
    const colors = generateColors(labels.length);
    
    const ctx = summaryChartCanvas.getContext('2d');
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.parsed.toFixed(2)}%`;
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Portfolio Allocation'
                }
            }
        }
    });
}

// Initialize portfolio performance chart
function initPortfolioPerformanceChart(portfolioData) {
    const performanceChartCanvas = document.getElementById('portfolioPerformanceChart');
    if (!performanceChartCanvas) return;
    
    const labels = [];
    const profitLossData = [];
    
    // Extract data for chart
    portfolioData.forEach(item => {
        labels.push(item.symbol);
        profitLossData.push(item.profit_loss_percent);
    });
    
    // Determine colors based on profit/loss
    const colors = profitLossData.map(value => 
        value >= 0 ? 'rgba(76, 175, 80, 0.8)' : 'rgba(244, 67, 54, 0.8)'
    );
    
    const ctx = performanceChartCanvas.getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Profit/Loss (%)',
                data: profitLossData,
                backgroundColor: colors,
                borderColor: colors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.parsed.x.toFixed(2)}%`;
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Portfolio Performance'
                }
            },
            scales: {
                x: {
                    grid: {
                        display: true
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

// Update portfolio table with formatted currency values
function updatePortfolioTable(portfolioData) {
    console.log('Updating portfolio table with data:', portfolioData);
    
    // Always remove loading state from portfolio table
    const portfolioTable = document.getElementById('portfolioTable');
    if (portfolioTable) {
        portfolioTable.classList.remove('loading');
    }
    
    // Try to find the table body with different selectors to ensure we find it
    let tableBody = document.getElementById('portfolioTableBody');
    
    if (!tableBody) {
        console.log('Could not find portfolioTableBody by ID, trying querySelector');
        tableBody = document.querySelector('#portfolioTableBody');
    }
    
    if (!tableBody) {
        console.log('Could not find portfolioTableBody by querySelector, trying tbody selector');
        tableBody = document.querySelector('#portfolioTable tbody');
    }
    
    // If we still can't find the table body, try to create it
    if (!tableBody && portfolioTable) {
        console.log('Creating new tbody element');
        tableBody = document.createElement('tbody');
        tableBody.id = 'portfolioTableBody';
        portfolioTable.appendChild(tableBody);
    }
    
    if (!tableBody) {
        console.error('Could not find or create portfolio table body element');
        // Try to find all tbody elements on the page to help with debugging
        const allTbodies = document.querySelectorAll('tbody');
        console.log('Found', allTbodies.length, 'tbody elements on the page:');
        allTbodies.forEach((tbody, index) => {
            console.log(`tbody ${index}:`, tbody);
        });
        return;
    }
    
    console.log('Found table body element:', tableBody);
    
    // Clear existing rows
    tableBody.innerHTML = '';
    
    if (!portfolioData || portfolioData.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="7" class="text-center">
                    <i class="fas fa-info-circle text-muted fa-2x mb-3"></i>
                    <p>No stocks in your portfolio. Add your first stock to get started!</p>
                    <a href="/portfolio/add" class="btn btn-sm btn-primary mt-2">
                        <i class="fas fa-plus-circle me-1"></i> Add Stock
                    </a>
                </td>
            </tr>
        `;
        return;
    }
    
    // Add new rows with formatted values
    portfolioData.forEach(item => {
        const row = document.createElement('tr');
        
        // Ensure all required values are present and are numbers
        const profitLoss = isNaN(parseFloat(item.profit_loss)) ? 0 : parseFloat(item.profit_loss);
        const profitLossPercent = isNaN(parseFloat(item.profit_loss_percent)) ? 0 : parseFloat(item.profit_loss_percent);
        const quantity = isNaN(parseFloat(item.quantity)) ? 0 : parseFloat(item.quantity);
        
        // Ensure current price and purchase price are valid numbers
        const currentPrice = isNaN(parseFloat(item.current_price)) ? 0 : parseFloat(item.current_price);
        const purchasePrice = isNaN(parseFloat(item.purchase_price)) ? 0 : parseFloat(item.purchase_price);
        
        // Calculate profit/loss class
        const profitLossClass = profitLoss >= 0 ? 'text-success' : 'text-danger';
        const profitLossSign = profitLoss >= 0 ? '+' : '';
        
        // Check if it's an Indian stock (NSE/BSE) or US stock (NASDAQ)
        const isIndianStock = item.symbol && (item.symbol.endsWith('.NS') || item.symbol.endsWith('.BO'));
        
        // Get stock exchange badge
        const exchangeBadge = isIndianStock ? 
            `<span class="badge bg-info">NSE/BSE</span>` : 
            `<span class="badge bg-primary">NASDAQ</span>`;
        
        // Use formatted values if available, otherwise format them safely
        const purchasePriceFormatted = item.purchase_price_formatted || formatCurrency(purchasePrice);
        const currentPriceFormatted = item.current_price_formatted || formatCurrency(currentPrice);
        const currentValueFormatted = item.current_value_formatted || formatCurrency(quantity * currentPrice);
        
        // Ensure we never display NaN in the UI
        const safeFormatCurrency = (value) => {
            if (isNaN(parseFloat(value)) || value === null || value === undefined) {
                return '₹0.00';
            }
            return formatCurrency(value);
        };
        
        // Get the profit/loss formatted value
        let profitLossFormatted;
        if (item.profit_loss_formatted && !item.profit_loss_formatted.includes('NaN')) {
            // Use the server-provided formatted value if it doesn't contain NaN
            profitLossFormatted = item.profit_loss_formatted;
        } else {
            // Format locally if server didn't provide formatted value or it contains NaN
            profitLossFormatted = `${profitLossSign}${safeFormatCurrency(Math.abs(profitLoss))}`;
        }
        
        // Format the row
        row.innerHTML = `
            <td>${item.symbol || 'Unknown'} ${exchangeBadge}</td>
            <td>${isNaN(quantity) ? '0.00' : quantity.toFixed(2)}</td>
            <td>${purchasePriceFormatted.includes('NaN') ? safeFormatCurrency(purchasePrice) : purchasePriceFormatted}</td>
            <td>${currentPriceFormatted.includes('NaN') ? safeFormatCurrency(currentPrice) : currentPriceFormatted}</td>
            <td>${currentValueFormatted.includes('NaN') ? safeFormatCurrency(quantity * currentPrice) : currentValueFormatted}</td>
            <td class="${profitLossClass}">${profitLossFormatted.includes('NaN') ? safeFormatCurrency(profitLoss) : profitLossFormatted}</td>
            <td>
                <div class="btn-group btn-group-sm">
                    <button class="btn btn-outline-primary view-stock-btn" data-symbol="${item.symbol || ''}">
                        <i class="fas fa-chart-line me-1"></i> View
                    </button>
                    <button class="btn btn-outline-danger sell-stock-btn" data-id="${item.id || ''}" data-symbol="${item.symbol || ''}" data-quantity="${isNaN(quantity) ? 0 : quantity}">
                        <i class="fas fa-dollar-sign me-1"></i> Sell
                    </button>
                </div>
            </td>
        `;
        
        tableBody.appendChild(row);
    });
    
    // Log for debugging
    console.log('Portfolio table updated with', portfolioData.length, 'items');
    
    // Add event listeners to view buttons
    document.querySelectorAll('.view-stock-btn').forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const symbol = this.dataset.symbol;
            if (symbol) {
                // Update the stock info container with the selected symbol
                const stockInfoContainer = document.getElementById('stock-info-container');
                if (stockInfoContainer) {
                    stockInfoContainer.dataset.symbol = symbol;
                    // Reinitialize the stock info display
                    if (typeof initStockInfoDisplay === 'function') {
                        initStockInfoDisplay();
                    } else if (typeof fetchStockData === 'function') {
                        fetchStockData(symbol);
                    }
                }
            }
        });
    });
    
    // Update top performers table
    updateTopPerformersTable(portfolioData);
}

// Update top performers table
function updateTopPerformersTable(portfolioData) {
    const topPerformersTable = document.getElementById('topPerformersTable');
    if (!topPerformersTable) return;
    
    // Clear existing rows
    topPerformersTable.innerHTML = '';
    
    // Ensure all items have the required properties with valid values
    const validData = portfolioData.filter(item => {
        return item && item.symbol && !isNaN(parseFloat(item.profit_loss_percent));
    });
    
    if (validData.length === 0) {
        // If no valid data, display a message
        const row = document.createElement('tr');
        row.innerHTML = `
            <td colspan="3" class="text-center">No performance data available</td>
        `;
        topPerformersTable.appendChild(row);
        return;
    }
    
    // Sort by profit/loss percent
    const sortedData = [...validData].sort((a, b) => {
        const aPercent = parseFloat(a.profit_loss_percent) || 0;
        const bPercent = parseFloat(b.profit_loss_percent) || 0;
        return bPercent - aPercent;
    });
    
    // Take top 3
    const topPerformers = sortedData.slice(0, 3);
    
    // Add rows
    topPerformers.forEach(item => {
        const row = document.createElement('tr');
        
        // Ensure values are numbers
        const profitLossPercent = parseFloat(item.profit_loss_percent) || 0;
        const profitLoss = parseFloat(item.profit_loss) || 0;
        
        // Determine classes and signs
        const percentClass = profitLossPercent >= 0 ? 'text-success' : 'text-danger';
        const percentSign = profitLossPercent >= 0 ? '+' : '';
        const valueClass = profitLoss >= 0 ? 'text-success' : 'text-danger';
        const valueSign = profitLoss >= 0 ? '+' : '';
        
        // Use formatted value if available, otherwise format it
        const formattedValue = item.profit_loss_formatted || formatCurrency(Math.abs(profitLoss));
        
        row.innerHTML = `
            <td>${item.symbol || 'Unknown'}</td>
            <td class="text-end ${percentClass}">
                ${percentSign}${profitLossPercent.toFixed(2)}%
            </td>
            <td class="text-end ${valueClass}">
                ${valueSign}${formattedValue}
            </td>
        `;
        topPerformersTable.appendChild(row);
    });
}

// Show empty portfolio message
function showEmptyPortfolioMessage(errorMessage = null) {
    const emptyMessage = document.getElementById('emptyPortfolioMessage');
    if (emptyMessage) {
        if (errorMessage) {
            emptyMessage.innerHTML = `
                <i class="fas fa-exclamation-circle me-2"></i> ${errorMessage}
                <a href="/portfolio/add" class="alert-link">Try adding a stock</a> to your portfolio.
            `;
        }
        emptyMessage.style.display = 'block';
    }
    
    // Hide portfolio sections
    const portfolioAnalysisCard = document.getElementById('portfolioAnalysisCard');
    const portfolioHoldingsCard = document.getElementById('portfolioHoldingsCard');
    
    if (portfolioAnalysisCard) portfolioAnalysisCard.style.display = 'none';
    if (portfolioHoldingsCard) portfolioHoldingsCard.style.display = 'none';
}

// Fetch recent transactions
function fetchRecentTransactions() {
    const transactionsTable = document.getElementById('recentTransactionsTable');
    if (!transactionsTable) {
        console.log('Recent transactions table not found');
        return;
    }
    
    console.log('Fetching recent transactions...');
    
    // Use AbortController to timeout fetch requests that take too long
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
    
    fetch('/portfolio/api/recent-transactions', {
        signal: controller.signal,
        headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
    })
    .then(response => {
        clearTimeout(timeoutId);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Recent transactions loaded:', data);
        updateRecentTransactionsTable(data.transactions);
    })
    .catch(error => {
        clearTimeout(timeoutId);
        console.error('Error fetching recent transactions:', error);
        showTransactionsError(transactionsTable, error.message);
    });
}

// Update recent transactions table
function updateRecentTransactionsTable(transactions) {
    const tableBody = document.getElementById('recentTransactionsTable');
    if (!tableBody) return;
    
    // Clear the table
    tableBody.innerHTML = '';
    
    if (!transactions || transactions.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="6" class="text-center">
                    <i class="fas fa-info-circle text-muted fa-2x mb-3"></i>
                    <p>No transactions found.</p>
                </td>
            </tr>
        `;
        return;
    }
    
    // Show only the 5 most recent transactions
    const recentTransactions = transactions.slice(0, 5);
    
    // Add each transaction to the table
    recentTransactions.forEach(transaction => {
        const row = document.createElement('tr');
        
        // Format the date
        const date = new Date(transaction.transaction_date);
        const formattedDate = date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
        
        // Determine badge class based on transaction type
        const badgeClass = transaction.transaction_type === 'buy' ? 'bg-success' : 'bg-danger';
        
        row.innerHTML = `
            <td>${formattedDate}</td>
            <td>${transaction.display_symbol || transaction.stock_symbol}</td>
            <td>
                <span class="badge ${badgeClass}">
                    ${transaction.transaction_type.toUpperCase()}
                </span>
            </td>
            <td>${transaction.quantity}</td>
            <td>${transaction.price_formatted}</td>
            <td>${transaction.total_formatted}</td>
        `;
        
        tableBody.appendChild(row);
    });
}

// Show error message in transactions table
function showTransactionsError(tableBody, errorMessage) {
    if (!tableBody) return;
    
    tableBody.innerHTML = `
        <tr>
            <td colspan="6" class="text-center">
                <i class="fas fa-exclamation-circle text-warning fa-2x mb-3"></i>
                <p>Error loading transactions: ${errorMessage}</p>
                <button class="btn btn-sm btn-outline-primary mt-2" onclick="fetchRecentTransactions()">
                    <i class="fas fa-sync-alt me-1"></i> Try Again
                </button>
            </td>
        </tr>
    `;
    document.getElementById('holdingsCountDisplay').innerHTML = '0';
}

// Setup sell button click handlers
function setupSellButtons() {
    // Add event listeners to sell buttons after portfolio data is loaded
    document.addEventListener('click', function(e) {
        if (e.target && e.target.classList.contains('sell-stock-btn')) {
            e.preventDefault();
            const id = e.target.dataset.id;
            const symbol = e.target.dataset.symbol;
            const quantity = e.target.dataset.quantity;
            
            if (id && symbol) {
                console.log(`Redirecting to sell page for ${symbol} (ID: ${id})`); 
                // Redirect to the sell page
                window.location.href = `/portfolio/sell/${id}`;
            }
        }
    });
}

// Calculate portfolio totals from portfolio items
function calculatePortfolioTotals(portfolioItems) {
    // Initialize totals
    let totalCost = 0;
    let totalValue = 0;
    let totalProfitLoss = 0;
    let totalProfitLossPercent = 0;
    
    // Calculate totals
    portfolioItems.forEach(item => {
        // Ensure values are numbers
        const cost = parseFloat(item.cost_basis) || 0;
        const value = parseFloat(item.current_value) || 0;
        
        totalCost += cost;
        totalValue += value;
    });
    
    // Calculate profit/loss
    totalProfitLoss = totalValue - totalCost;
    
    // Calculate profit/loss percentage (avoid division by zero)
    if (totalCost > 0) {
        totalProfitLossPercent = (totalProfitLoss / totalCost) * 100;
    }
    
    // Format the values for display
    return {
        total_cost: totalCost,
        total_cost_formatted: formatCurrency(totalCost),
        total_value: totalValue,
        total_value_formatted: formatCurrency(totalValue),
        total_profit_loss: totalProfitLoss,
        total_profit_loss_formatted: formatCurrency(totalProfitLoss),
        total_profit_loss_percent: totalProfitLossPercent
    };
}

// Generate colors for charts
function generateColors(count) {
    const colors = [
        'rgba(255, 99, 132, 0.8)',
        'rgba(54, 162, 235, 0.8)',
        'rgba(255, 206, 86, 0.8)',
        'rgba(75, 192, 192, 0.8)',
        'rgba(153, 102, 255, 0.8)',
        'rgba(255, 159, 64, 0.8)',
        'rgba(255, 99, 132, 0.8)',
        'rgba(54, 162, 235, 0.8)',
        'rgba(255, 206, 86, 0.8)',
        'rgba(75, 192, 192, 0.8)'
    ];
    
    // If we need more colors than we have, just repeat them
    if (count > colors.length) {
        const extraNeeded = count - colors.length;
        for (let i = 0; i < extraNeeded; i++) {
            colors.push(colors[i % colors.length]);
        }
    }
    
    return colors.slice(0, count);
}

// Format currency with INR symbol
function formatCurrency(value) {
    if (value === undefined || value === null) {
        return '₹0.00';
    }
    
    // Ensure value is a number
    const numValue = parseFloat(value);
    if (isNaN(numValue)) {
        return '₹0.00';
    }
    
    // Format with Indian Rupee symbol and thousands separators
    return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(numValue);
}

// Setup add stock form validation
function setupAddStockForm() {
    const addStockForm = document.getElementById('addStockForm');
    if (!addStockForm) return;
    
    addStockForm.addEventListener('submit', function(event) {
        if (!this.checkValidity()) {
            event.preventDefault();
            event.stopPropagation();
        }
        
        this.classList.add('was-validated');
    });
}

// Setup sell stock form validation
function setupSellStockForm() {
    const sellStockForm = document.getElementById('sellStockForm');
    if (!sellStockForm) return;
    
    // Get the maximum quantity allowed
    const maxQuantity = parseFloat(sellStockForm.getAttribute('data-max-quantity') || 0);
    
    // Get the quantity input field
    const quantityInput = document.getElementById('quantity');
    if (quantityInput) {
        // Set the max attribute
        quantityInput.setAttribute('max', maxQuantity);
        
        // Add input validation
        quantityInput.addEventListener('input', function() {
            const value = parseFloat(this.value) || 0;
            if (value > maxQuantity) {
                this.setCustomValidity(`You only have ${maxQuantity} shares to sell`);
            } else if (value <= 0) {
                this.setCustomValidity('Quantity must be greater than 0');
            } else {
                this.setCustomValidity('');
            }
        });
    }
    
    // Form submission validation
    sellStockForm.addEventListener('submit', function(event) {
        // Check if the form is valid
        if (!this.checkValidity()) {
            event.preventDefault();
            event.stopPropagation();
            
            // Show validation messages
            const invalidFields = this.querySelectorAll(':invalid');
            if (invalidFields.length > 0) {
                // Focus the first invalid field
                invalidFields[0].focus();
            }
        } else {
            // Form is valid, show a confirmation dialog
            const quantity = parseFloat(quantityInput.value) || 0;
            const sellPrice = parseFloat(document.getElementById('sell_price').value) || 0;
            const totalValue = quantity * sellPrice;
            
            if (!confirm(`Are you sure you want to sell ${quantity} shares for a total of ${formatCurrency(totalValue)}?`)) {
                event.preventDefault();
            }
        }
        
        this.classList.add('was-validated');
    });
}

// Update stock information component with a stock from the portfolio
function updateStockInfoComponent(portfolioData) {
    const stockInfoContainer = document.getElementById('stock-info-container');
    if (!stockInfoContainer) return;
    
    // Sort by profit/loss percent to find the best performing stock
    const sortedData = [...portfolioData].sort((a, b) => b.profit_loss_percent - a.profit_loss_percent);
    
    // Get the best performing stock or the first one if no positive performers
    const selectedStock = sortedData[0];
    
    if (selectedStock) {
        // Set the symbol in the container's data attribute
        stockInfoContainer.dataset.symbol = selectedStock.symbol;
        
        // Initialize the stock info display
        // This will trigger the fetch in stock-info.js
        if (typeof initStockInfoDisplay === 'function') {
            initStockInfoDisplay();
        }
    }
}