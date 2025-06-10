// Live stock ticker functionality

// Wait for DOM to load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize ticker if element exists
    const tickerContainer = document.querySelector('.ticker-container');
    if (tickerContainer) {
        initLiveStockTicker();
    }
});

// Initialize live stock ticker
function initLiveStockTicker() {
    // Define default symbols to display in ticker
    const defaultSymbols = [
        'SPY', 'QQQ', 'DIA', 'IWM',           // Market indices ETFs
        'AAPL', 'MSFT', 'GOOGL', 'AMZN',      // Tech giants
        'FB', 'TSLA', 'NFLX', 'NVDA',         // More tech stocks
        'JPM', 'BAC', 'WFC', 'GS',            // Banking stocks
        'JNJ', 'PFE', 'MRK', 'ABT',           // Healthcare stocks
        'XOM', 'CVX', 'COP', 'BP'             // Energy stocks
    ];
    
    // Get stored symbols or use defaults
    let tickerSymbols = localStorage.getItem('tickerSymbols');
    tickerSymbols = tickerSymbols ? tickerSymbols.split(',') : defaultSymbols;
    
    // Initial ticker load
    updateTickerData(tickerSymbols);
    
    // Update ticker every 60 seconds
    setInterval(() => {
        updateTickerData(tickerSymbols);
    }, 60000);
    
    // Setup ticker symbols customization
    setupTickerCustomization(tickerSymbols);
}

// Update ticker data with current prices
function updateTickerData(symbols) {
    const tickerTape = document.querySelector('.ticker-tape');
    if (!tickerTape) return;
    
    // Show loading indicator
    tickerTape.innerHTML = '<div class="ticker-item">Loading ticker data...</div>';
    
    // Symbols list as string for API
    const symbolsStr = symbols.join(',');
    
    // Fetch data from API
    fetch(`/api/ticker-data?symbols=${symbolsStr}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            let tickerHtml = '';
            
            // Process each stock
            symbols.forEach(symbol => {
                if (data[symbol]) {
                    const stockData = data[symbol];
                    const priceClass = stockData.percent_change >= 0 ? 'stock-change-positive' : 'stock-change-negative';
                    const priceSign = stockData.percent_change >= 0 ? '+' : '';
                    
                    tickerHtml += `
                        <div class="ticker-item" data-symbol="${symbol}">
                            <span class="stock-symbol">${symbol}</span>
                            <span class="stock-price">₹${stockData.price.toFixed(2)}</span>
                            <span class="${priceClass}">
                                ${priceSign}${stockData.percent_change.toFixed(2)}%
                            </span>
                        </div>
                    `;
                } else {
                    // If data not available for this symbol
                    tickerHtml += `
                        <div class="ticker-item" data-symbol="${symbol}">
                            <span class="stock-symbol">${symbol}</span>
                            <span class="stock-price">N/A</span>
                        </div>
                    `;
                }
            });
            
            // Update ticker content
            tickerTape.innerHTML = tickerHtml;
            
            // Make ticker items clickable to view details
            const tickerItems = document.querySelectorAll('.ticker-item');
            tickerItems.forEach(item => {
                item.addEventListener('click', function() {
                    const symbol = this.dataset.symbol;
                    window.location.href = `/prediction?symbol=${symbol}`;
                });
            });
        })
        .catch(error => {
            console.error('Error fetching ticker data:', error);
            tickerTape.innerHTML = '<div class="ticker-item">Error loading ticker data. Please try again later.</div>';
        });
}

// Setup ticker symbols customization modal
function setupTickerCustomization(currentSymbols) {
    const customizeButton = document.getElementById('customizeTicker');
    if (!customizeButton) return;
    
    // Handle button click to open customization modal
    customizeButton.addEventListener('click', function() {
        // Create modal dynamically
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.id = 'tickerCustomizeModal';
        modal.setAttribute('tabindex', '-1');
        modal.setAttribute('aria-labelledby', 'tickerCustomizeModalLabel');
        modal.setAttribute('aria-hidden', 'true');
        
        modal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="tickerCustomizeModalLabel">Customize Ticker Symbols</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <p>Enter stock symbols separated by commas (e.g., AAPL, MSFT, GOOGL)</p>
                        <textarea class="form-control" id="tickerSymbolsInput" rows="4">${currentSymbols.join(', ')}</textarea>
                        <div class="mt-3">
                            <h6>Popular Symbols</h6>
                            <div class="popular-symbols">
                                <button class="btn btn-sm btn-outline-primary m-1" data-symbol="SPY">SPY</button>
                                <button class="btn btn-sm btn-outline-primary m-1" data-symbol="QQQ">QQQ</button>
                                <button class="btn btn-sm btn-outline-primary m-1" data-symbol="AAPL">AAPL</button>
                                <button class="btn btn-sm btn-outline-primary m-1" data-symbol="MSFT">MSFT</button>
                                <button class="btn btn-sm btn-outline-primary m-1" data-symbol="GOOGL">GOOGL</button>
                                <button class="btn btn-sm btn-outline-primary m-1" data-symbol="AMZN">AMZN</button>
                                <button class="btn btn-sm btn-outline-primary m-1" data-symbol="FB">FB</button>
                                <button class="btn btn-sm btn-outline-primary m-1" data-symbol="TSLA">TSLA</button>
                                <button class="btn btn-sm btn-outline-primary m-1" data-symbol="NFLX">NFLX</button>
                                <button class="btn btn-sm btn-outline-primary m-1" data-symbol="JPM">JPM</button>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" id="saveTickerSymbols">Save Changes</button>
                    </div>
                </div>
            </div>
        `;
        
        // Add modal to document
        document.body.appendChild(modal);
        
        // Initialize Bootstrap modal
        const modalInstance = new bootstrap.Modal(modal);
        modalInstance.show();
        
        // Add click handlers for popular symbols
        const popularButtons = modal.querySelectorAll('.popular-symbols button');
        popularButtons.forEach(button => {
            button.addEventListener('click', function() {
                const symbol = this.dataset.symbol;
                const textarea = document.getElementById('tickerSymbolsInput');
                const currentText = textarea.value;
                
                // Check if symbol is already in the list
                if (!currentText.includes(symbol)) {
                    textarea.value = currentText ? `${currentText}, ${symbol}` : symbol;
                }
            });
        });
        
        // Handle save button
        const saveButton = document.getElementById('saveTickerSymbols');
        saveButton.addEventListener('click', function() {
            const textarea = document.getElementById('tickerSymbolsInput');
            const symbolsText = textarea.value;
            
            // Parse and clean up symbols
            let symbols = symbolsText.split(/[ ,]+/)
                .map(s => s.trim().toUpperCase())
                .filter(s => s.length > 0);
            
            // Remove duplicates
            symbols = [...new Set(symbols)];
            
            // Limit number of symbols to prevent performance issues
            if (symbols.length > 30) {
                symbols = symbols.slice(0, 30);
                alert('Maximum 30 symbols allowed. We\'ve kept the first 30 symbols from your list.');
            }
            
            // Save to localStorage
            localStorage.setItem('tickerSymbols', symbols.join(','));
            
            // Update ticker
            updateTickerData(symbols);
            
            // Close modal
            modalInstance.hide();
            
            // Remove modal from DOM after hiding
            modal.addEventListener('hidden.bs.modal', function() {
                document.body.removeChild(modal);
            });
        });
        
        // Clean up modal when hidden
        modal.addEventListener('hidden.bs.modal', function() {
            document.body.removeChild(modal);
        });
    });
}
