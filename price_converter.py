"""
Price converter utility for StockSage application.
Handles all currency conversions and ensures consistent price display across the application.
"""

import logging
import os
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache

# Default exchange rate to use if API is unavailable
DEFAULT_USD_TO_INR_RATE = 84.45

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
    if not symbol:
        return False
    # Indian stocks have .NS or .BO suffix
    return symbol.endswith('.NS') or symbol.endswith('.BO')

def is_us_stock(symbol):
    """Check if a stock symbol is from a US exchange"""
    return not is_indian_stock(symbol)

def convert_usd_to_inr(usd_value):
    """Convert a USD value to INR"""
    rate = get_usd_to_inr_rate()
    return usd_value * rate

def format_price_as_inr(price):
    """Format a price value as INR with the ₹ symbol"""
    try:
        return f"₹{float(price):,.2f}"  # Added comma separator for thousands
    except (ValueError, TypeError):
        return "₹0.00"

def convert_to_inr(symbol, price):
    """Convert a price to INR based on the stock symbol
    
    Args:
        symbol: The stock symbol to check if it's Indian or US
        price: The price value to convert if needed
        
    Returns:
        Price in INR
    """
    # Ensure price is a float
    try:
        price = float(price)
    except (TypeError, ValueError):
        return 0.0
        
    # Convert to INR if it's a US stock
    if not is_indian_stock(symbol):
        return convert_usd_to_inr(price)
    else:
        return price  # Already in INR

def display_price(symbol, price):
    """Format a price value with the appropriate currency symbol based on stock origin
    
    Args:
        symbol: The stock symbol to check if it's Indian or US
        price: The price value to format
        
    Returns:
        Formatted price string with INR symbol
    """
    # Ensure price is a float
    try:
        price = float(price)
    except (TypeError, ValueError):
        return "₹0.00"
        
    # Convert to INR if it's a US stock
    inr_price = convert_to_inr(symbol, price)
    
    # Format with INR symbol and thousand separators
    return f"₹{inr_price:,.2f}"

def format_currency(price, force_inr=False, include_sign=False):
    """Format a price value with the appropriate currency symbol
    
    Args:
        price: The price value to format
        force_inr: Force INR formatting regardless of symbol
        include_sign: Include + sign for positive values
        
    Returns:
        Formatted price string with currency symbol
    """
    # Ensure price is a float
    try:
        price = float(price)
    except (TypeError, ValueError):
        return "₹0.00"
    
    # Determine sign
    sign = ""
    if include_sign and price > 0:
        sign = "+"
    elif price < 0:
        sign = "-"
        price = abs(price)  # Use absolute value for formatting
    
    # Format with INR symbol and thousand separators
    return f"{sign}₹{price:,.2f}"

def convert_to_inr(symbol, price):
    """Convert a price to INR if necessary based on the stock symbol"""
    # Handle None or invalid price
    if price is None or not isinstance(price, (int, float)):
        return 0.0
        
    if is_us_stock(symbol):
        return price * get_usd_to_inr_rate()
    return price  # Already in INR

# Alias for backward compatibility
ensure_inr_price = convert_to_inr

def convert_stock_prices(prices, symbol):
    """Convert a list of stock prices to INR if needed"""
    if is_us_stock(symbol):
        return [convert_to_inr(symbol, price) for price in prices]
    return prices

def convert_prediction_prices(predictions, symbol):
    """Convert prediction prices to INR if needed and add formatted price strings"""
    for prediction in predictions:
        if 'price' in prediction:
            # Store original USD price for reference if it's a US stock
            if is_us_stock(symbol):
                prediction['price_usd'] = prediction['price']
                prediction['price_usd_formatted'] = f"${prediction['price_usd']:,.2f}"
                # Convert USD to INR
                prediction['price'] = convert_to_inr(symbol, prediction['price'])
                
            # Add formatted price string with currency symbol
            prediction['price_formatted'] = display_price(symbol, prediction['price'])
    
    return predictions
