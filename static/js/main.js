// Main JavaScript file for the application

// Wait for DOM to load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Stock Market Prediction Web App initialized');
    
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize Bootstrap popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Check for flash messages and display using toast
    const flashMessages = document.querySelectorAll('.flash-message');
    if (flashMessages.length > 0) {
        flashMessages.forEach(function(message) {
            const toastEl = document.createElement('div');
            toastEl.className = `toast align-items-center text-white bg-${message.dataset.category} border-0`;
            toastEl.setAttribute('role', 'alert');
            toastEl.setAttribute('aria-live', 'assertive');
            toastEl.setAttribute('aria-atomic', 'true');
            
            const toastContent = `
                <div class="d-flex">
                    <div class="toast-body">
                        ${message.textContent}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
            `;
            
            toastEl.innerHTML = toastContent;
            document.querySelector('.toast-container').appendChild(toastEl);
            
            const toast = new bootstrap.Toast(toastEl, {
                autohide: true,
                delay: 5000
            });
            toast.show();
        });
    }
    
    // Initialize ticker tape
    initTickerTape();
    
    // Add event listeners for stock lookup forms
    const stockLookupForms = document.querySelectorAll('.stock-lookup-form');
    if (stockLookupForms.length > 0) {
        stockLookupForms.forEach(form => {
            form.addEventListener('submit', function(e) {
                e.preventDefault();
                const symbolInput = this.querySelector('input[name="symbol"]');
                const symbol = symbolInput.value.trim().toUpperCase();
                if (symbol) {
                    fetchStockData(symbol);
                }
            });
        });
    }
    
    // Initialize theme switcher
    initThemeSwitcher();
    
    // Initialize any stock information containers
    initStockInfoContainers();
    
    // Initialize comment system
    initCommentSystem();
    
    // Format currency numbers
    formatCurrencyNumbers();
});

// Initialize live stock ticker at top of page
function initTickerTape() {
    const tickerContainer = document.querySelector('.ticker-tape');
    if (!tickerContainer) return;
    
    // Default symbols
    const symbols = 'SPY,QQQ,AAPL,MSFT,GOOGL,AMZN,FB,TSLA,NFLX,NVDA';
    
    // Fetch ticker data
    fetchTickerData(symbols);
    
    // Refresh ticker every 60 seconds
    setInterval(() => {
        fetchTickerData(symbols);
    }, 60000);
}

// Fetch stock ticker data from API
function fetchTickerData(symbols) {
    const tickerContainer = document.querySelector('.ticker-tape');
    if (!tickerContainer) return;
    
    fetch(`/api/ticker-data?symbols=${symbols}`)
        .then(response => response.json())
        .then(data => {
            let tickerHtml = '';
            
            for (const symbol in data) {
                const stockData = data[symbol];
                const changeClass = stockData.percent_change >= 0 ? 'stock-change-positive' : 'stock-change-negative';
                const changeSign = stockData.percent_change >= 0 ? '+' : '';
                
                tickerHtml += `
                    <div class="ticker-item">
                        <span class="stock-symbol">${symbol}</span>
                        <span class="stock-price">₹${stockData.price.toFixed(2)}</span>
                        <span class="${changeClass}">
                            ${changeSign}${stockData.percent_change.toFixed(2)}%
                        </span>
                    </div>
                `;
            }
            
            tickerContainer.innerHTML = tickerHtml;
        })
        .catch(error => {
            console.error('Error fetching ticker data:', error);
            tickerContainer.innerHTML = '<div class="ticker-item">Error loading ticker data</div>';
        });
}

// Initialize theme switcher functionality
function initThemeSwitcher() {
    const themeSwitch = document.getElementById('themeSwitch');
    const themeIcon = document.getElementById('themeIcon');
    const themeLabel = document.getElementById('themeLabel');
    if (!themeSwitch) return;
    
    // Check for saved theme preference or default to light theme
    const savedTheme = localStorage.getItem('theme') || 'light';
    
    // Apply the theme
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
        themeSwitch.checked = true;
        updateThemeUI(true);
    } else {
        // Ensure light theme is applied
        document.body.classList.remove('dark-theme'); 
        themeSwitch.checked = false;
        updateThemeUI(false);
    }
    
    // Listen for theme switch changes
    themeSwitch.addEventListener('change', function() {
        if (this.checked) {
            // Switch to dark theme
            document.body.classList.add('dark-theme');
            localStorage.setItem('theme', 'dark');
            updateThemeUI(true);
        } else {
            // Switch to light theme
            document.body.classList.remove('dark-theme');
            localStorage.setItem('theme', 'light');
            updateThemeUI(false);
        }
    });
    
    // Helper function to update theme UI elements
    function updateThemeUI(isDark) {
        if (themeIcon) {
            themeIcon.className = isDark ? 'fas fa-moon' : 'fas fa-sun';
        }
        if (themeLabel) {
            themeLabel.textContent = isDark ? 'Dark Mode' : 'Light Mode';
        }
    }
}

// Initialize comment system for sentiment analysis
function initCommentSystem() {
    const commentForm = document.getElementById('commentForm');
    if (!commentForm) return;
    
    commentForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const commentInput = document.getElementById('commentInput');
        const stockSymbol = this.dataset.symbol;
        
        if (!commentInput.value.trim()) {
            showAlert('Please enter a comment', 'danger');
            return;
        }
        
        // Submit comment via AJAX
        fetch('/sentiment/comment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: JSON.stringify({
                symbol: stockSymbol,
                comment: commentInput.value
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Add new comment to the list
                const commentsList = document.getElementById('commentsList');
                const newComment = createCommentElement(data.comment);
                commentsList.prepend(newComment);
                commentInput.value = '';
                
                showAlert('Comment added successfully', 'success');
            } else {
                showAlert(data.error || 'Error adding comment', 'danger');
            }
        })
        .catch(error => {
            console.error('Error adding comment:', error);
            showAlert('Error adding comment', 'danger');
        });
    });
    
    // Delegation for reply buttons
    document.addEventListener('click', function(e) {
        if (e.target && e.target.classList.contains('reply-button')) {
            const commentId = e.target.dataset.commentId;
            const replyForm = document.getElementById(`replyForm-${commentId}`);
            
            if (replyForm.style.display === 'none' || !replyForm.style.display) {
                replyForm.style.display = 'block';
            } else {
                replyForm.style.display = 'none';
            }
        }
    });
    
    // Delegation for reply form submission
    document.addEventListener('submit', function(e) {
        if (e.target && e.target.classList.contains('reply-form')) {
            e.preventDefault();
            
            const commentId = e.target.dataset.commentId;
            const replyInput = document.getElementById(`replyInput-${commentId}`);
            
            if (!replyInput.value.trim()) {
                showAlert('Please enter a reply', 'danger');
                return;
            }
            
            // Submit reply via AJAX
            fetch('/sentiment/reply', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: JSON.stringify({
                    comment_id: commentId,
                    reply: replyInput.value
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Add new reply to the list
                    const repliesList = document.getElementById(`replies-${commentId}`);
                    const newReply = createReplyElement(data.reply);
                    repliesList.appendChild(newReply);
                    replyInput.value = '';
                    
                    showAlert('Reply added successfully', 'success');
                } else {
                    showAlert(data.error || 'Error adding reply', 'danger');
                }
            })
            .catch(error => {
                console.error('Error adding reply:', error);
                showAlert('Error adding reply', 'danger');
            });
        }
    });
}

// Create a comment HTML element
function createCommentElement(comment) {
    const commentDiv = document.createElement('div');
    commentDiv.className = 'comment-container';
    commentDiv.id = `comment-${comment.id}`;
    
    commentDiv.innerHTML = `
        <div class="comment">
            <div class="comment-header">
                <span class="comment-username">${comment.username}</span>
                <span class="comment-timestamp">${comment.created_at}</span>
            </div>
            <div class="comment-body">
                ${comment.comment}
            </div>
            <div class="comment-footer mt-2">
                <span class="badge bg-${comment.sentiment === 'positive' ? 'success' : (comment.sentiment === 'negative' ? 'danger' : 'warning')}">
                    ${comment.sentiment.toUpperCase()}
                </span>
                <button class="btn btn-sm btn-outline-primary reply-button" data-comment-id="${comment.id}">Reply</button>
            </div>
            <div class="reply-form-container mt-3" style="display: none;" id="replyForm-${comment.id}">
                <form class="reply-form" data-comment-id="${comment.id}">
                    <div class="input-group">
                        <input type="text" class="form-control" id="replyInput-${comment.id}" placeholder="Write a reply...">
                        <button type="submit" class="btn btn-primary">Reply</button>
                    </div>
                </form>
            </div>
        </div>
        <div class="comment-replies" id="replies-${comment.id}">
            <!-- Replies will be added here -->
        </div>
    `;
    
    return commentDiv;
}

// Create a reply HTML element
function createReplyElement(reply) {
    const replyDiv = document.createElement('div');
    replyDiv.className = 'reply';
    replyDiv.id = `reply-${reply.id}`;
    
    replyDiv.innerHTML = `
        <div class="reply-header">
            <span class="reply-username">${reply.username}</span>
            <span class="reply-timestamp">${reply.created_at}</span>
        </div>
        <div class="reply-body">
            ${reply.reply}
        </div>
    `;
    
    return replyDiv;
}

// Show an alert message
function showAlert(message, type) {
    const alertContainer = document.getElementById('alertContainer');
    if (!alertContainer) return;
    
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.setAttribute('role', 'alert');
    
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertContainer.appendChild(alertDiv);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        alertDiv.classList.remove('show');
        setTimeout(() => {
            alertDiv.remove();
        }, 150);
    }, 5000);
}

// Format numbers as currency
function formatCurrencyNumbers() {
    const currencyElements = document.querySelectorAll('.currency');
    if (!currencyElements.length) return;
    
    currencyElements.forEach(element => {
        const value = parseFloat(element.textContent);
        if (!isNaN(value)) {
            element.textContent = new Intl.NumberFormat('en-IN', {
                style: 'currency',
                currency: 'INR'
            }).format(value);
        }
    });
}

/**
 * Fetch stock data from the API for a given symbol
 * Handles both NSE and NASDAQ stocks
 * @param {string} symbol - The stock symbol to fetch data for
 */
function fetchStockData(symbol) {
    // Show loading state
    const stockInfoContainer = document.getElementById('stock-info-container');
    if (stockInfoContainer) {
        stockInfoContainer.innerHTML = `
            <div class="text-center p-4">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">Fetching data for ${symbol}...</p>
            </div>
        `;
        stockInfoContainer.classList.remove('d-none');
    }
    
    // Fetch data from our API
    fetch(`/api/stock-data/${symbol}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                displayStockData(data, stockInfoContainer);
            } else {
                displayStockError(data.error || 'Unknown error', stockInfoContainer);
            }
        })
        .catch(error => {
            console.error('Error fetching stock data:', error);
            displayStockError(error.message, stockInfoContainer);
        });
}

/**
 * Display stock data in the UI
 * @param {Object} data - The stock data from the API
 * @param {HTMLElement} container - The container element to display the data in
 */
function displayStockData(data, container) {
    if (!container) return;
    
    // Check if data has the expected structure
    if (!data || !data.price) {
        console.error('Invalid stock data structure:', data);
        displayStockError('Invalid stock data received', container);
        return;
    }
    
    // Determine which price to display based on stock origin
    let priceDisplay = '';
    if (data.is_indian) {
        priceDisplay = data.price.formatted_inr || `₹${data.price.current.toFixed(2)}`;
    } else {
        const inrPrice = data.price.formatted_inr || `₹${data.price.current_inr.toFixed(2)}`;
        const usdPrice = data.price.formatted_usd || `$${data.price.current.toFixed(2)}`;
        priceDisplay = `${inrPrice} <small class="text-muted">(${usdPrice})</small>`;
    }
    
    // Determine the change color
    const changeValue = data.change && data.change.value !== undefined ? data.change.value : 0;
    const changeClass = changeValue >= 0 ? 'text-success' : 'text-danger';
    const changeIcon = changeValue >= 0 ? 
        '<i class="bi bi-arrow-up-circle-fill"></i>' : 
        '<i class="bi bi-arrow-down-circle-fill"></i>';
    
    // Create the HTML for the stock info
    const html = `
        <div class="card shadow-sm">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="mb-0">${data.symbol || ''} ${data.modified_symbol ? `(${data.modified_symbol})` : ''}</h5>
                ${data.is_mock_data ? '<span class="badge bg-warning">Using Mock Data</span>' : ''}
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h2 class="mb-0">${priceDisplay}</h2>
                        <p class="${changeClass}">
                            ${changeIcon} ${data.change.formatted}
                        </p>
                        <p class="text-muted small">Last updated: ${data.date}</p>
                    </div>
                    <div class="col-md-6 text-md-end mt-3 mt-md-0">
                        <div class="btn-group">
                            <a href="/portfolio/add-stock?symbol=${data.symbol}" class="btn btn-primary">
                                <i class="bi bi-plus-circle"></i> Add to Portfolio
                            </a>
                            <button type="button" class="btn btn-outline-secondary" onclick="fetchHistoricalData('${data.symbol}')">
                                <i class="bi bi-graph-up"></i> Chart
                            </button>
                        </div>
                    </div>
                </div>
                
                <hr>
                
                <div class="row mt-3">
                    <div class="col-6">
                        <p><strong>Exchange:</strong> ${data.is_indian ? 'Indian (NSE/BSE)' : 'US (NASDAQ/NYSE)'}</p>
                    </div>
                    <div class="col-6">
                        <p><strong>Volume:</strong> ${data.volume.toLocaleString()}</p>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

/**
 * Display an error message when stock data fetch fails
 * @param {HTMLElement} container - The container element to display the error in
 * @param {string} error - The error message
 */
function displayStockError(error, container) {
    if (!container) return;
    
    // Instead of showing an error message, clear the container
    container.innerHTML = '';
    
    // Log the error to console for debugging
    console.log('Stock error (suppressed from UI):', error);
    
    // Optional: Add a small, non-intrusive message or icon
    // that doesn't use the alert-danger class
    container.innerHTML = `
        <div class="text-center p-3">
            <button class="btn btn-sm btn-outline-primary" onclick="window.location.reload()">
                <i class="fas fa-sync-alt me-1"></i> Refresh Data
            </button>
        </div>
    `;
}

/**
 * Initialize stock information containers with pre-loaded symbols
 */
function initStockInfoContainers() {
    const containers = document.querySelectorAll('[data-stock-symbol]');
    containers.forEach(container => {
        const symbol = container.dataset.stockSymbol;
        if (symbol) {
            // Set the container ID so we can reference it later
            container.id = 'stock-info-container';
            fetchStockData(symbol);
        }
    });
}
