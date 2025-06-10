// Theme switcher functionality

document.addEventListener('DOMContentLoaded', function() {
    console.log('Theme switcher initialized');
    
    // Get theme switch element
    const themeSwitch = document.getElementById('themeSwitch');
    if (!themeSwitch) return;
    
    // Initialize theme based on user's saved preference
    initializeTheme();
    
    // Add event listener to toggle theme when switch is toggled
    themeSwitch.addEventListener('change', toggleTheme);
    
    // Add keyboard accessibility
    themeSwitch.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            this.checked = !this.checked;
            toggleTheme();
        }
    });
});

// Initialize theme based on user's saved preference or system preference
function initializeTheme() {
    const themeSwitch = document.getElementById('themeSwitch');
    if (!themeSwitch) return;
    
    // First check if we have a saved preference
    let savedTheme = localStorage.getItem('theme');
    
    // If no saved preference, check for system preference
    if (!savedTheme) {
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            savedTheme = 'dark';
        } else {
            savedTheme = 'light';
        }
    }
    
    // Apply theme
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
        themeSwitch.checked = true;
        updateThemeIcon(true);
    } else {
        document.body.classList.remove('dark-theme');
        themeSwitch.checked = false;
        updateThemeIcon(false);
    }
}

// Toggle theme function
function toggleTheme() {
    const themeSwitch = document.getElementById('themeSwitch');
    if (!themeSwitch) return;
    
    const isDarkMode = themeSwitch.checked;
    
    if (isDarkMode) {
        document.body.classList.add('dark-theme');
        localStorage.setItem('theme', 'dark');
    } else {
        document.body.classList.remove('dark-theme');
        localStorage.setItem('theme', 'light');
    }
    
    updateThemeIcon(isDarkMode);
    
    // Reload any charts to update their appearance
    updateChartsForTheme(isDarkMode);
}

// Update theme icon based on current mode
function updateThemeIcon(isDarkMode) {
    const themeIcon = document.getElementById('themeIcon');
    if (!themeIcon) return;
    
    if (isDarkMode) {
        themeIcon.className = 'fas fa-moon';
        document.getElementById('themeLabel').textContent = 'Dark Mode';
    } else {
        themeIcon.className = 'fas fa-sun';
        document.getElementById('themeLabel').textContent = 'Light Mode';
    }
}

// Update charts when theme changes
function updateChartsForTheme(isDarkMode) {
    if (typeof Chart === 'undefined') return;
    
    // Force update all chart instances
    Chart.helpers.each(Chart.instances, function(chart) {
        // Logic to update chart colors if needed
        chart.update();
    });
}

// Listen for system theme changes
if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
        const isDarkMode = e.matches;
        const themeSwitch = document.getElementById('themeSwitch');
        if (!themeSwitch) return;
        
        // Only update if user hasn't set a preference
        if (!localStorage.getItem('theme')) {
            themeSwitch.checked = isDarkMode;
            toggleTheme();
        }
    });
}
