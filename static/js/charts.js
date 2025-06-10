// Charts and data visualization functionality

// Wait for DOM to load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Charts.js initialized');
    
    // Initialize stock price chart if present
    initStockPriceChart();
    
    // Initialize portfolio composition chart if present
    initPortfolioChart();
    
    // Initialize sentiment analysis chart if present
    initSentimentChart();
    
    // Initialize prediction models comparison chart if present
    initPredictionChart();
});

// Initialize stock price chart
function initStockPriceChart() {
    const stockChartCanvas = document.getElementById('stockPriceChart');
    if (!stockChartCanvas) return;
    
    // Get data from the HTML element
    const chartData = JSON.parse(stockChartCanvas.dataset.chartData);
    
    // Create the chart
    const ctx = stockChartCanvas.getContext('2d');
    const stockChart = new Chart(ctx, {
        type: 'line',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    usePointStyle: true,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            // Check if this is an Indian stock by looking at the dataset label
                            const isIndianStock = label.includes('.NS') || label.includes('.BO');
                            
                            // Format the price in the appropriate currency
                            label += new Intl.NumberFormat(isIndianStock ? 'en-IN' : 'en-US', {
                                style: 'currency',
                                currency: isIndianStock ? 'INR' : 'USD'
                            }).format(context.parsed.y);
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Price (USD)'
                    },
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

// Initialize portfolio composition chart
function initPortfolioChart() {
    const portfolioChartCanvas = document.getElementById('portfolioCompositionChart');
    if (!portfolioChartCanvas) return;
    
    // Fetch portfolio data
    fetch('/portfolio/api/portfolio-data')
        .then(response => response.json())
        .then(data => {
            if (!data.composition || Object.keys(data.composition).length === 0) {
                // No portfolio data
                const noDataContainer = document.createElement('div');
                noDataContainer.className = 'text-center py-5';
                noDataContainer.innerHTML = '<p class="mb-0">No portfolio data available. Add stocks to your portfolio.</p>';
                portfolioChartCanvas.parentNode.replaceChild(noDataContainer, portfolioChartCanvas);
                return;
            }
            
            // Prepare data for chart
            const labels = Object.keys(data.composition);
            const values = Object.values(data.composition);
            const backgroundColors = generateChartColors(labels.length);
            
            // Create chart
            const ctx = portfolioChartCanvas.getContext('2d');
            const portfolioChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        data: values,
                        backgroundColor: backgroundColors
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: {
                                padding: 20
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `${context.label}: ${context.raw.toFixed(2)}%`;
                                }
                            }
                        }
                    }
                }
            });
        })
        .catch(error => {
            console.error('Error fetching portfolio data for chart:', error);
            const errorContainer = document.createElement('div');
            errorContainer.className = 'alert alert-danger';
            errorContainer.textContent = 'Error loading portfolio data for chart.';
            portfolioChartCanvas.parentNode.replaceChild(errorContainer, portfolioChartCanvas);
        });
}

// Initialize sentiment analysis chart
function initSentimentChart() {
    const sentimentChartCanvas = document.getElementById('sentimentChart');
    if (!sentimentChartCanvas) return;
    
    // Get data attributes
    const positive = parseInt(sentimentChartCanvas.dataset.positive) || 0;
    const negative = parseInt(sentimentChartCanvas.dataset.negative) || 0;
    const neutral = parseInt(sentimentChartCanvas.dataset.neutral) || 0;
    
    // Create chart
    const ctx = sentimentChartCanvas.getContext('2d');
    const sentimentChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Positive', 'Negative', 'Neutral'],
            datasets: [{
                data: [positive, negative, neutral],
                backgroundColor: [
                    'rgba(76, 175, 80, 0.8)',
                    'rgba(244, 67, 54, 0.8)',
                    'rgba(255, 152, 0, 0.8)'
                ]
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
                            const total = positive + negative + neutral;
                            const percentage = total > 0 ? ((context.raw / total) * 100).toFixed(1) : 0;
                            return `${context.label}: ${context.raw} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Initialize prediction models comparison chart
function initPredictionChart() {
    const predictionChartCanvas = document.getElementById('predictionChart');
    if (!predictionChartCanvas) return;
    
    // Get prediction data
    const predictions = JSON.parse(predictionChartCanvas.dataset.predictions);
    const symbol = predictionChartCanvas.dataset.symbol;
    
    if (Object.keys(predictions).length === 0) {
        const noDataContainer = document.createElement('div');
        noDataContainer.className = 'alert alert-info';
        noDataContainer.textContent = 'No prediction data available for this stock.';
        predictionChartCanvas.parentNode.replaceChild(noDataContainer, predictionChartCanvas);
        return;
    }
    
    // Organize data for chart
    const datasets = [];
    let labels = [];
    
    // Different colors for different models
    const colors = {
        'Linear Regression': 'rgba(54, 162, 235, 0.8)',
        'ARIMA': 'rgba(75, 192, 192, 0.8)',
        'LSTM': 'rgba(153, 102, 255, 0.8)'
    };
    
    // Process each model's predictions
    for (const model in predictions) {
        const modelData = predictions[model];
        const data = [];
        const dates = [];
        
        for (const prediction of modelData.predictions) {
            data.push(prediction.price);
            dates.push(prediction.date);
        }
        
        // For the first model, set the labels (dates)
        if (labels.length === 0) {
            labels = dates;
        }
        
        datasets.push({
            label: `${model} (${(modelData.confidence * 100).toFixed(1)}% confidence)`,
            data: data,
            borderColor: colors[model] || 'rgba(255, 99, 132, 0.8)',
            backgroundColor: colors[model] || 'rgba(255, 99, 132, 0.8)',
            tension: 0.1
        });
    }
    
    // Create the chart
    const ctx = predictionChartCanvas.getContext('2d');
    const predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            // Check if this is an Indian stock by looking at the dataset label
                            const isIndianStock = label.includes('.NS') || label.includes('.BO');
                            
                            // Format the price in the appropriate currency
                            label += new Intl.NumberFormat(isIndianStock ? 'en-IN' : 'en-US', {
                                style: 'currency',
                                currency: isIndianStock ? 'INR' : 'USD'
                            }).format(context.parsed.y);
                            return label;
                        }
                    }
                },
                title: {
                    display: true,
                    text: `${symbol} Price Predictions`
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Price (USD)'
                    },
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

// Generate chart colors based on count
function generateChartColors(count) {
    const baseColors = [
        'rgba(75, 192, 192, 0.8)',
        'rgba(54, 162, 235, 0.8)',
        'rgba(153, 102, 255, 0.8)',
        'rgba(255, 159, 64, 0.8)',
        'rgba(255, 99, 132, 0.8)',
        'rgba(255, 205, 86, 0.8)',
        'rgba(201, 203, 207, 0.8)',
        'rgba(100, 181, 246, 0.8)',
        'rgba(129, 199, 132, 0.8)',
        'rgba(239, 83, 80, 0.8)'
    ];
    
    // If we need more colors than in base array, generate them
    if (count <= baseColors.length) {
        return baseColors.slice(0, count);
    }
    
    // Generate additional colors
    const colors = [...baseColors];
    
    for (let i = baseColors.length; i < count; i++) {
        const r = Math.floor(Math.random() * 255);
        const g = Math.floor(Math.random() * 255);
        const b = Math.floor(Math.random() * 255);
        colors.push(`rgba(${r}, ${g}, ${b}, 0.8)`);
    }
    
    return colors;
}
