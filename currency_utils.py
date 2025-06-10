"""
Currency conversion utilities for StockSage application.
Ensures consistent currency handling across the application.
"""

import logging
import os
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache

# Default exchange rate to use if API is unavailable
DEFAULT_USD_TO_INR_RATE = 83.5

def ensure_cache_dir():
    """Ensure the cache directory exists"""
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

def get_exchange_rate_cache_path():
    """Get the path to the exchange rate cache file"""
    cache_dir = ensure_cache_dir()
    return cache_dir / "exchange_rates.json"

def save_exchange_rates_to_cache(rates):
    """Save exchange rates to cache"""
    cache_path = get_exchange_rate_cache_path()
    try:
        data = {
            'rates': rates,
            'timestamp': datetime.now().timestamp()
        }
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        logging.info("Cached exchange rates")
    except Exception as e:
        logging.error(f"Error caching exchange rates: {e}")

def load_exchange_rates_from_cache(max_age_hours=24):
    """Load exchange rates from cache if not too old"""
    cache_path = get_exchange_rate_cache_path()
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
        
        # Check if cache is too old
        timestamp = data.get('timestamp', 0)
        age_hours = (datetime.now().timestamp() - timestamp) / 3600
        if age_hours > max_age_hours:
            logging.info(f"Exchange rate cache is {age_hours:.1f} hours old, refreshing")
            return None
        
        logging.info("Using cached exchange rates")
        return data.get('rates', {})
    except Exception as e:
        logging.error(f"Error loading exchange rate cache: {e}")
    
    return None

def fetch_exchange_rates():
    """Fetch latest exchange rates from API"""
    # Try to load from cache first
    cached_rates = load_exchange_rates_from_cache()
    if cached_rates and 'USD_INR' in cached_rates:
        return cached_rates
    
    # If not in cache, fetch from API
    api_key = os.environ.get('EXCHANGE_RATE_API_KEY')
    if not api_key:
        logging.warning("No Exchange Rate API key found, using default rate")
        return {'USD_INR': DEFAULT_USD_TO_INR_RATE}
    
    try:
        url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/USD"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('result') == 'success':
                rates = data.get('conversion_rates', {})
                usd_to_inr = rates.get('INR', DEFAULT_USD_TO_INR_RATE)
                result = {'USD_INR': usd_to_inr}
                save_exchange_rates_to_cache(result)
                logging.info(f"Fetched exchange rates: USD to INR = {usd_to_inr}")
                return result
    except Exception as e:
        logging.error(f"Error fetching exchange rates: {e}")
    
    # If API call fails, use default rate
    logging.warning("Using default USD to INR exchange rate")
    return {'USD_INR': DEFAULT_USD_TO_INR_RATE}

@lru_cache(maxsize=1)
def get_usd_to_inr_rate():
    """Get the current USD to INR exchange rate with caching"""
    rates = fetch_exchange_rates()
    return rates.get('USD_INR', DEFAULT_USD_TO_INR_RATE)

def is_indian_stock(symbol):
    """Check if a stock symbol is from an Indian exchange"""
    return symbol.endswith('.NS') or symbol.endswith('.BO')

def convert_usd_to_inr(usd_value):
    """Convert a USD value to INR"""
    rate = get_usd_to_inr_rate()
    return usd_value * rate

def format_price_as_inr(price):
    """Format a price value as INR with the ₹ symbol and proper thousands separators"""
    # Use the locale-aware formatting to ensure proper thousands separators
    return f"₹{price:,.2f}"

def ensure_inr_price(price, symbol):
    """Ensure a price is in INR, converting if necessary"""
    if not is_indian_stock(symbol):
        return convert_usd_to_inr(price)
    return price
