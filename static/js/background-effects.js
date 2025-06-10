// Background effects and interactive elements for StockSage
document.addEventListener('DOMContentLoaded', function() {
    // Add scrolled class to navbar on scroll
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 50) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        });
    }

    // Add positive/negative classes to ticker items based on content
    const tickerItems = document.querySelectorAll('.ticker-item');
    if (tickerItems.length > 0) {
        tickerItems.forEach(item => {
            const text = item.textContent.trim();
            if (text.includes('+')) {
                item.classList.add('positive');
            } else if (text.includes('-')) {
                item.classList.add('negative');
            }
        });
    }

    // Add subtle hover effects to stock elements
    const stockElements = document.querySelectorAll('.stock-price, .stock-symbol, .portfolio-value');
    if (stockElements.length > 0) {
        stockElements.forEach(element => {
            element.addEventListener('mouseenter', function() {
                this.style.transform = 'scale(1.05)';
                this.style.transition = 'all 0.3s ease';
            });
            
            element.addEventListener('mouseleave', function() {
                this.style.transform = 'scale(1)';
            });
        });
    }
});
