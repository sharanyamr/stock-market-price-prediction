"""
Cache configuration for the StockSage application.
This module provides a simple in-memory cache to reduce API calls and database load.
"""

import functools
import time
from typing import Dict, Any, Callable, Tuple

# Cache storage
_CACHE: Dict[str, Tuple[float, Any]] = {}
DEFAULT_CACHE_TIMEOUT = 300  # 5 minutes in seconds


def cached(timeout: int = DEFAULT_CACHE_TIMEOUT):
    """
    Decorator to cache function results for a specified time period.
    
    Args:
        timeout: Cache timeout in seconds (default: 5 minutes)
    
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from function name and arguments
            skip_cache = kwargs.pop('skip_cache', False)
            cache_key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            
            # Check if result is in cache and not expired
            current_time = time.time()
            if not skip_cache and cache_key in _CACHE:
                timestamp, result = _CACHE[cache_key]
                if current_time - timestamp < timeout:
                    return result
            
            # Call the function and cache the result
            result = func(*args, **kwargs)
            _CACHE[cache_key] = (current_time, result)
            return result
        return wrapper
    return decorator


def clear_cache():
    """Clear the entire cache."""
    global _CACHE
    _CACHE = {}


def clear_expired_cache():
    """Remove expired items from cache."""
    current_time = time.time()
    global _CACHE
    _CACHE = {
        key: (timestamp, value) 
        for key, (timestamp, value) in _CACHE.items()
        if current_time - timestamp < DEFAULT_CACHE_TIMEOUT
    }


def get_cache_stats():
    """Return statistics about the cache."""
    return {
        "total_items": len(_CACHE),
        "memory_usage_estimate": sum(len(str(v)) for _, v in _CACHE.values()) / 1024,  # KB
    }
