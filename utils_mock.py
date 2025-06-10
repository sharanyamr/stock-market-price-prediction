import pandas as pd
from datetime import datetime, timedelta
import hashlib
import random
import logging

def get_mock_stock_data(symbol, period='1mo'):
    """Generate mock stock data when Yahoo Finance API is unavailable
    
    Args:
        symbol (str): Stock ticker symbol
        period (str): Time period for historical data
        
    Returns:
        DataFrame: Mock historical stock data
    """
    logging.info(f"Generating mock data for {symbol} with period {period}")
    
    # Create a date range for the period
    today = datetime.now()
    
    if period == '1d':
        days = 1
    elif period == '5d':
        days = 5
    elif period == '1mo':
        days = 30
    elif period == '3mo':
        days = 90
    elif period == '6mo':
        days = 180
    elif period == '1y':
        days = 365
    else:
        days = 30  # Default to 1 month
    
    # Generate dates
    dates = [today - timedelta(days=i) for i in range(days, -1, -1)]
    
    # Generate a base price (random but deterministic for the same symbol)
    hash_obj = hashlib.md5(symbol.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    base_price = (hash_int % 1000) + 100  # Between 100 and 1100
    
    # Generate price data with some randomness but following a trend
    random.seed(hash_int)  # Make it deterministic for the same symbol
    
    # Determine if it's an upward or downward trend
    trend = random.choice([1, -1])  # 1 for upward, -1 for downward
    
    # Generate prices
    prices = []
    volumes = []
    opens = []
    highs = []
    lows = []
    
    current_price = base_price
    for i in range(len(dates)):
        # Add some randomness to the price
        daily_change = random.uniform(-0.02, 0.02)  # -2% to +2%
        trend_change = trend * random.uniform(0, 0.01)  # 0 to 1% in the trend direction
        change = current_price * (daily_change + trend_change)
        
        # Calculate today's price
        current_price += change
        
        # Make sure price doesn't go below a minimum
        current_price = max(current_price, 10)
        
        # Calculate high, low, open
        daily_high = current_price * (1 + random.uniform(0, 0.01))  # 0 to 1% higher
        daily_low = current_price * (1 - random.uniform(0, 0.01))   # 0 to 1% lower
        daily_open = random.uniform(daily_low, daily_high)  # Open somewhere between high and low
        
        # Generate a random volume
        volume = int(random.uniform(100000, 10000000))  # Between 100K and 10M
        
        prices.append(current_price)
        volumes.append(volume)
        opens.append(daily_open)
        highs.append(daily_high)
        lows.append(daily_low)
    
    # Create a DataFrame
    data = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volumes
    }, index=dates)
    
    # Mark this as mock data
    if hasattr(data, 'attrs'):
        data.attrs['mock_data'] = True
    
    return data
