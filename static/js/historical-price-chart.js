// Historical price chart using Chart.js
document.addEventListener('DOMContentLoaded', function() {
    // Get the chart data from the template
    const chartData = window.chartData;
    
    if (!chartData || !chartData.labels || !chartData.datasets) {
        console.error('Chart data is missing or invalid');
        return;
    }
    
    // Get the canvas element
    const ctx = document.getElementById('historicalPriceChart');
    
    if (!ctx) {
        console.error('Canvas element not found');
        return;
    }
    
    // Create the chart
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.labels,
            datasets: [{
                label: chartData.datasets[0].label,
                data: chartData.datasets[0].data,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderWidth: 2,
                pointRadius: 3,
                pointBackgroundColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                // Check if it's an Indian stock to determine currency symbol
                                const isIndianStock = document.body.hasAttribute('data-is-indian-stock') ? 
                                    document.body.getAttribute('data-is-indian-stock') === 'true' : false;
                                
                                const currencySymbol = isIndianStock ? '₹' : '$';
                                label += currencySymbol + context.parsed.y.toFixed(2);
                            }
                            return label;
                        }
                    }
                },
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Historical Price Chart'
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Date'
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Price'
                    }
                }
            }
        }
    });
});
