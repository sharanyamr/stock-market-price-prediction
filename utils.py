import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import requests
import os

from extensions import db
from models import NewsArticle

import time
import random
import json
import hashlib
from functools import lru_cache
from pathlib import Path

# Import mock data generator
from utils_mock import get_mock_stock_data

# Import our currency utilities
from currency_utils import (
    is_indian_stock, 
    convert_usd_to_inr, 
    format_price_as_inr, 
    ensure_inr_price,
    get_usd_to_inr_rate
)

# Create a cache directory if it doesn't exist
def ensure_cache_dir():
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

# Cache for stock data to reduce API calls
def get_cache_path(symbol, period, exchange=None):
    cache_dir = ensure_cache_dir()
    if exchange:
        cache_file = f"{symbol}_{exchange}_{period}.json"
    else:
        cache_file = f"{symbol}_{period}.json"
    return cache_dir / cache_file

def save_to_cache(data, symbol, period, exchange=None):
    if data.empty:
        return
    
    cache_path = get_cache_path(symbol, period, exchange)
    try:
        # Convert DataFrame to JSON
        json_data = {
            'index': [str(idx) for idx in data.index],
            'data': data.to_dict(orient='records'),
            'timestamp': datetime.now().timestamp(),
            'symbol': symbol,
            'period': period,
            'exchange': exchange
        }
        with open(cache_path, 'w') as f:
            json.dump(json_data, f)
        logging.info(f"Cached data for {symbol} with period {period}")
    except Exception as e:
        logging.error(f"Error caching data: {e}")

def load_from_cache(symbol, period, exchange=None, max_age_hours=4):
    cache_path = get_cache_path(symbol, period, exchange)
    if not cache_path.exists():
        return pd.DataFrame()
    
    try:
        with open(cache_path, 'r') as f:
            json_data = json.load(f)
        
        # Check if cache is too old
        timestamp = json_data.get('timestamp', 0)
        age_hours = (datetime.now().timestamp() - timestamp) / 3600
        if age_hours > max_age_hours:
            logging.info(f"Cache for {symbol} is {age_hours:.1f} hours old, refreshing")
            return pd.DataFrame()
        
        # Convert JSON back to DataFrame
        df = pd.DataFrame(json_data['data'])
        if not df.empty:
            # Convert index back to datetime
            df.index = pd.to_datetime(json_data['index'])
            logging.info(f"Loaded cached data for {symbol} with period {period}")
            return df
    except Exception as e:
        logging.error(f"Error loading cache: {e}")
    
    return pd.DataFrame()

def fetch_stock_data(symbol, period='1mo', exchange=None, use_cache=True):
    """Fetch historical stock data with rate limiting, caching, and better error handling
    
    Args:
        symbol (str): Stock ticker symbol
        period (str): Time period for historical data
        exchange (str, optional): Specific exchange to use ('NSE', 'BSE', 'NASDAQ')
        use_cache (bool): Whether to use cached data if available
        
    Returns:
        DataFrame: Historical stock data
    """
    # Log the request details
    logging.info(f"Fetching stock data for symbol: {symbol}, period: {period}, exchange: {exchange}")
    # First check cache if enabled
    if use_cache:
        cached_data = load_from_cache(symbol, period, exchange)
        if not cached_data.empty:
            return cached_data
    
    # Add random delay to avoid hitting rate limits (0.5 to 2 seconds)
    delay = random.uniform(0.5, 2.0)
    time.sleep(delay)
    
    try:
        # Handle exchange-specific formatting
        original_symbol = symbol
        formatted_symbol = symbol
        
        # If exchange is explicitly specified, format accordingly
        if exchange:
            if exchange.upper() == 'NSE' and not symbol.endswith('.NS'):
                formatted_symbol = f"{symbol}.NS"
            elif exchange.upper() == 'BSE' and not symbol.endswith('.BO'):
                formatted_symbol = f"{symbol}.BO"
            # NASDAQ and other US exchanges don't need suffix
            
            logging.info(f"Using {exchange} formatted symbol: {formatted_symbol}")
        
        # First try to use real data from Yahoo Finance
        try:
            # Create a ticker object
            stock = yf.Ticker(formatted_symbol)
            
            # Use a try-except block for each API call
            data = stock.history(period=period)
            if not data.empty:
                logging.info(f"Successfully fetched real data for {formatted_symbol}")
                save_to_cache(data, symbol, period, exchange)
                return data
            else:
                logging.warning(f"No data returned for {formatted_symbol} with period {period}")
        except Exception as e:
            logging.error(f"Error fetching data for {formatted_symbol}: {e}")
            if '429' in str(e):
                # Rate limit hit, add additional delay
                logging.warning("Rate limit hit, adding additional delay")
                time.sleep(random.uniform(2.0, 5.0))
            
        # If we get here, the real data fetch failed, so use mock data
        logging.warning(f"Using mock data for {formatted_symbol} due to API failure")
        
        # Add a flag to indicate this is mock data
        mock_data = get_mock_stock_data(formatted_symbol, period)
        
        # Add metadata to the mock data
        mock_data.attrs['is_mock'] = True
        mock_data.attrs['original_symbol'] = symbol
        mock_data.attrs['formatted_symbol'] = formatted_symbol
        mock_data.attrs['exchange'] = exchange
        
        # Save to cache with a shorter expiration (2 hours for mock data)
        save_to_cache(mock_data, symbol, period, exchange)
        return mock_data
            
    except Exception as e:
        # If we get here, something went very wrong
        logging.error(f"Unexpected error in fetch_stock_data: {e}")
        
        # Try to return cached data even if it's old as a fallback
        fallback_data = load_from_cache(symbol, period, exchange, max_age_hours=48)
        if not fallback_data.empty:
            logging.info(f"Using older cached data as fallback for {symbol}")
            return fallback_data
        
        # If all else fails, use mock data
        logging.warning(f"Using mock data as the final fallback for {symbol}")
        mock_data = get_mock_stock_data(symbol, period)
        
        # Add metadata to the mock data
        mock_data.attrs['is_mock'] = True
        mock_data.attrs['original_symbol'] = symbol
        mock_data.attrs['formatted_symbol'] = formatted_symbol if 'formatted_symbol' in locals() else symbol
        mock_data.attrs['exchange'] = exchange
        mock_data.attrs['is_fallback'] = True
        
        return mock_data

def fetch_latest_news(count=20):
    """Fetch latest stock market news from API"""
    try:
        NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
        
        # Try to use News API first
        if NEWS_API_KEY and NEWS_API_KEY != 'd0dgn0pr01qhd5a0c390d0dgn0pr01qhd5a0c39g':  # Check if it's not the placeholder
            try:
                url = 'https://newsapi.org/v2/everything'
                params = {
                    'q': 'stock market OR finance OR investing',
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': count,
                    'apiKey': NEWS_API_KEY
                }
                
                logging.info(f"Fetching news from News API with key: {NEWS_API_KEY[:4]}...")
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                news_data = response.json()
                
                if news_data['status'] == 'ok' and len(news_data.get('articles', [])) > 0:
                    logging.info(f"Successfully fetched {len(news_data['articles'])} articles from News API")
                    return process_news_api_articles(news_data['articles'])
                else:
                    logging.warning(f"News API returned status: {news_data.get('status')} with message: {news_data.get('message')}")
            except Exception as e:
                logging.error(f"Error using News API: {e}")
        else:
            logging.warning("Valid News API key not found in environment variables")
        
        # If News API fails or no key, try alternative news sources
        return fetch_alternative_news_sources(count)
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error fetching news: {e}")
        return 0


def process_news_api_articles(articles):
    """Process and save articles from News API"""
    try:
        saved_count = 0
        current_time = datetime.now()
        
        for article in articles:
            # Skip articles without URLs or with example domains
            if not article.get('url') or 'example.com' in article.get('url', ''):
                continue
                
            # Check if article already exists
            existing_article = NewsArticle.query.filter_by(url=article['url']).first()
            
            if not existing_article:
                try:
                    # Parse date
                    published_at = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                except Exception:
                    published_at = current_time
                
                # Determine category based on content
                category = determine_article_category(article.get('title', ''), article.get('description', ''))
                
                # Create new article
                news_article = NewsArticle(
                    title=article.get('title', 'No Title'),
                    content=article.get('description', '') or article.get('content', '') or 'No description available',
                    source=article.get('source', {}).get('name', 'Unknown Source'),
                    url=article['url'],
                    published_at=published_at,
                    fetched_at=current_time,
                    category=category
                )
                
                db.session.add(news_article)
                saved_count += 1
        
        db.session.commit()
        return saved_count
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error processing news articles: {e}")
        return 0


def determine_article_category(title, description):
    """Determine the category of a news article based on its content"""
    title_desc = (title + ' ' + description).lower()
    
    if any(term in title_desc for term in ['stock', 'market', 'index', 'dow', 's&p', 'nasdaq', 'nse', 'bse']):
        return 'Markets'
    elif any(term in title_desc for term in ['tech', 'technology', 'software', 'hardware', 'ai', 'artificial intelligence']):
        return 'Technology'
    elif any(term in title_desc for term in ['economy', 'economic', 'gdp', 'growth', 'recession', 'inflation']):
        return 'Economy'
    elif any(term in title_desc for term in ['oil', 'gas', 'gold', 'silver', 'commodity', 'commodities']):
        return 'Commodities'
    elif any(term in title_desc for term in ['crypto', 'bitcoin', 'ethereum', 'blockchain']):
        return 'Cryptocurrency'
    else:
        return 'General'


def fetch_alternative_news_sources(count=20):
    """Fetch news from alternative sources when News API is not available"""
    try:
        # Try to fetch from Yahoo Finance RSS feed
        try:
            import feedparser
            import html
            
            yahoo_feed_url = 'https://finance.yahoo.com/news/rssindex'
            feed = feedparser.parse(yahoo_feed_url)
            
            if feed.entries and len(feed.entries) > 0:
                logging.info(f"Successfully fetched {len(feed.entries)} articles from Yahoo Finance RSS")
                
                saved_count = 0
                current_time = datetime.now()
                
                for entry in feed.entries[:count]:
                    # Check if article already exists
                    existing_article = NewsArticle.query.filter_by(url=entry.link).first()
                    
                    if not existing_article:
                        # Parse date
                        try:
                            published_at = datetime(*entry.published_parsed[:6])
                        except Exception:
                            published_at = current_time
                        
                        # Clean description (remove HTML tags)
                        content = html.unescape(entry.description)
                        
                        # Create new article
                        news_article = NewsArticle(
                            title=entry.title,
                            content=content,
                            source='Yahoo Finance',
                            url=entry.link,
                            published_at=published_at,
                            fetched_at=current_time,
                            category=determine_article_category(entry.title, content)
                        )
                        
                        db.session.add(news_article)
                        saved_count += 1
                
                db.session.commit()
                return saved_count
        except Exception as e:
            logging.error(f"Error fetching from Yahoo Finance RSS: {e}")
        
        # If all else fails, use sample news but with real URLs
        logging.warning("Using sample news with real URLs as fallback")
        current_time = datetime.now()
        sample_articles = [
            {
                'title': 'Markets Rally as Fed Signals Potential Rate Cut',
                'content': 'Stock markets surged today after the Federal Reserve indicated it might consider rate cuts later this year if inflation continues to moderate. The S&P 500 closed up 1.8% while the Nasdaq gained over 2.3%.',
                'source': {'name': 'CNBC'},
                'url': 'https://www.cnbc.com/markets/',
                'publishedAt': (current_time - timedelta(hours=2)).isoformat(),
                'category': 'Markets'
            },
            {
                'title': 'Tech Stocks Lead the Way in Market Rebound',
                'content': 'Technology stocks are leading a market rebound as strong earnings reports from major players suggest the sector remains resilient despite economic uncertainties. Semiconductor stocks particularly outperformed with gains exceeding 3%.',
                'source': {'name': 'Bloomberg'},
                'url': 'https://www.bloomberg.com/markets/stocks/technology',
                'publishedAt': (current_time - timedelta(hours=5)).isoformat(),
                'category': 'Technology'
            },
            {
                'title': 'Investor Sentiment Improves as Inflation Data Shows Signs of Cooling',
                'content': 'The latest inflation data showed signs of moderation, improving investor sentiment. Consumer prices rose at a slower pace than expected, potentially giving the Federal Reserve more flexibility in its monetary policy decisions.',
                'source': {'name': 'Reuters'},
                'url': 'https://www.reuters.com/markets/us/',
                'publishedAt': (current_time - timedelta(hours=8)).isoformat(),
                'category': 'Economy'
            },
            {
                'title': 'Oil Prices Rise Amid Supply Concerns',
                'content': 'Oil prices climbed today as supply concerns overshadowed demand worries. Brent crude futures rose above $85 per barrel after a major producer announced unexpected maintenance at key facilities.',
                'source': {'name': 'Financial Times'},
                'url': 'https://www.ft.com/commodities',
                'publishedAt': (current_time - timedelta(hours=10)).isoformat(),
                'category': 'Commodities'
            },
            {
                'title': 'Small Cap Stocks Outperform as Economic Outlook Improves',
                'content': 'Small-cap stocks outperformed their larger counterparts today as investors grew more optimistic about economic growth prospects. The Russell 2000 index gained over 2%, outpacing the S&P 500.',
                'source': {'name': 'Wall Street Journal'},
                'url': 'https://www.wsj.com/market-data',
                'publishedAt': (current_time - timedelta(hours=12)).isoformat(),
                'category': 'Markets'
            }               
        ]
        
        # Save sample news articles to database
        saved_count = 0
        for article in sample_articles:
            # Check if article already exists
            existing = NewsArticle.query.filter_by(url=article['url']).first()
            if existing:
                continue
                
            # Create new article
            news = NewsArticle(
                title=article['title'],
                content=article['content'],
                source=article['source']['name'],
                url=article['url'],
                published_at=datetime.fromisoformat(article['publishedAt']),
                fetched_at=current_time,
                category=article['category']
            )
            
            db.session.add(news)
            saved_count += 1
            
        try:
            db.session.commit()
            logging.info(f"Added {saved_count} sample news articles with real URLs")
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error saving sample news: {e}")
            
        return saved_count
        
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': 'stock market OR finance OR investing',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': count,
            'apiKey': NEWS_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        news_data = response.json()
        
        if news_data['status'] != 'ok':
            logging.error(f"News API error: {news_data.get('message', 'Unknown error')}")
            return 0
        
        articles = news_data['articles']
        saved_count = 0
        
        for article in articles:
            # Check if article already exists
            existing_article = NewsArticle.query.filter_by(url=article['url']).first()
            
            if not existing_article:
                # Parse date
                published_at = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                
                # Create new article
                news_article = NewsArticle(
                    title=article['title'],
                    content=article['description'] or 'No description available',
                    source=article['source']['name'],
                    url=article['url'],
                    published_at=published_at
                )
                
                db.session.add(news_article)
                saved_count += 1
        
        db.session.commit()
        return saved_count
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error fetching news: {e}")
        return 0

def get_currency_exchange_rate(from_currency, to_currency):
    """Get currency exchange rate from API"""
    try:
        EXCHANGE_API_KEY = os.environ.get('EXCHANGE_API_KEY')
        
        if not EXCHANGE_API_KEY:
            logging.warning("Exchange API key not found in environment variables")
            return None
        
        url = f'https://api.exchangerate-api.com/v4/latest/{from_currency}'
        
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        rates = data.get('rates', {})
        
        if to_currency in rates:
            return rates[to_currency]
        else:
            logging.error(f"Currency {to_currency} not found in exchange rates")
            return None
    except Exception as e:
        logging.error(f"Error getting exchange rate: {e}")
        return None

def get_bse_tickers():
    """Get a list of BSE (Bombay Stock Exchange) tickers
    
    Returns:
        DataFrame: BSE tickers with symbol and company name
    """
    try:
        # Try to fetch from a reliable source
        # For now, we'll return a list of popular BSE stocks
        popular_bse = [
            {'Symbol': 'RELIANCE.BO', 'Name': 'Reliance Industries Ltd.'},
            {'Symbol': 'TCS.BO', 'Name': 'Tata Consultancy Services Ltd.'},
            {'Symbol': 'HDFCBANK.BO', 'Name': 'HDFC Bank Ltd.'},
            {'Symbol': 'INFY.BO', 'Name': 'Infosys Ltd.'},
            {'Symbol': 'HINDUNILVR.BO', 'Name': 'Hindustan Unilever Ltd.'},
            {'Symbol': 'ICICIBANK.BO', 'Name': 'ICICI Bank Ltd.'},
            {'Symbol': 'KOTAKBANK.BO', 'Name': 'Kotak Mahindra Bank Ltd.'},
            {'Symbol': 'BAJFINANCE.BO', 'Name': 'Bajaj Finance Ltd.'},
            {'Symbol': 'BHARTIARTL.BO', 'Name': 'Bharti Airtel Ltd.'},
            {'Symbol': 'SBIN.BO', 'Name': 'State Bank of India'},
            {'Symbol': 'ASIANPAINT.BO', 'Name': 'Asian Paints Ltd.'},
            {'Symbol': 'MARUTI.BO', 'Name': 'Maruti Suzuki India Ltd.'},
            {'Symbol': 'ITC.BO', 'Name': 'ITC Ltd.'},
            {'Symbol': 'AXISBANK.BO', 'Name': 'Axis Bank Ltd.'},
            {'Symbol': 'LT.BO', 'Name': 'Larsen & Toubro Ltd.'},
            {'Symbol': 'HCLTECH.BO', 'Name': 'HCL Technologies Ltd.'},
            {'Symbol': 'WIPRO.BO', 'Name': 'Wipro Ltd.'},
            {'Symbol': 'ULTRACEMCO.BO', 'Name': 'UltraTech Cement Ltd.'},
            {'Symbol': 'SUNPHARMA.BO', 'Name': 'Sun Pharmaceutical Industries Ltd.'},
            {'Symbol': 'TITAN.BO', 'Name': 'Titan Company Ltd.'},
            {'Symbol': 'BAJAJFINSV.BO', 'Name': 'Bajaj Finserv Ltd.'},
            {'Symbol': 'TECHM.BO', 'Name': 'Tech Mahindra Ltd.'},
            {'Symbol': 'NESTLEIND.BO', 'Name': 'Nestle India Ltd.'},
            {'Symbol': 'TATASTEEL.BO', 'Name': 'Tata Steel Ltd.'},
            {'Symbol': 'POWERGRID.BO', 'Name': 'Power Grid Corporation of India Ltd.'},
            {'Symbol': 'NTPC.BO', 'Name': 'NTPC Ltd.'},
            {'Symbol': 'BAJAJ-AUTO.BO', 'Name': 'Bajaj Auto Ltd.'},
            {'Symbol': 'HDFCLIFE.BO', 'Name': 'HDFC Life Insurance Company Ltd.'},
            {'Symbol': 'M&M.BO', 'Name': 'Mahindra & Mahindra Ltd.'},
            {'Symbol': 'TATAMOTORS.BO', 'Name': 'Tata Motors Ltd.'}
        ]
        return pd.DataFrame(popular_bse)
        
        # Create DataFrame from the manual list
        ticker_df = pd.DataFrame(tickers)
        logging.info(f"Created a list of {len(ticker_df)} popular BSE tickers")
        return ticker_df
    except Exception as e:
        logging.error(f"Error creating BSE tickers: {e}")
        return pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Exchange'])


def get_nasdaq_tickers():
    """Get list of NASDAQ tickers"""
    try:
        # First try the GitHub source
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.csv"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            ticker_df = pd.read_csv(pd.io.common.StringIO(response.text))
            if not ticker_df.empty:
                # Add a column to indicate exchange
                ticker_df['Exchange'] = 'NASDAQ'
                logging.info(f"Successfully fetched {len(ticker_df)} NASDAQ tickers from GitHub")
                return ticker_df
                
        # If GitHub source failed, try an alternative source
        url = "https://www.nasdaq.com/market-activity/stocks/screener"
        logging.info("Using NASDAQ screener alternative method")
        
        # Create a basic list of popular NASDAQ tickers
        tickers = [
            {"Symbol": "AAPL", "Name": "Apple Inc.", "Sector": "Technology", "Exchange": "NASDAQ"},
            {"Symbol": "MSFT", "Name": "Microsoft Corporation", "Sector": "Technology", "Exchange": "NASDAQ"},
            {"Symbol": "AMZN", "Name": "Amazon.com Inc.", "Sector": "Consumer Cyclical", "Exchange": "NASDAQ"},
            {"Symbol": "GOOGL", "Name": "Alphabet Inc.", "Sector": "Communication Services", "Exchange": "NASDAQ"},
            {"Symbol": "META", "Name": "Meta Platforms Inc.", "Sector": "Communication Services", "Exchange": "NASDAQ"},
            {"Symbol": "TSLA", "Name": "Tesla Inc.", "Sector": "Consumer Cyclical", "Exchange": "NASDAQ"},
            {"Symbol": "NVDA", "Name": "NVIDIA Corporation", "Sector": "Technology", "Exchange": "NASDAQ"},
            {"Symbol": "PYPL", "Name": "PayPal Holdings Inc.", "Sector": "Financial Services", "Exchange": "NASDAQ"},
            {"Symbol": "NFLX", "Name": "Netflix Inc.", "Sector": "Communication Services", "Exchange": "NASDAQ"},
            {"Symbol": "INTC", "Name": "Intel Corporation", "Sector": "Technology", "Exchange": "NASDAQ"},
            {"Symbol": "CMCSA", "Name": "Comcast Corporation", "Sector": "Communication Services", "Exchange": "NASDAQ"},
            {"Symbol": "PEP", "Name": "PepsiCo Inc.", "Sector": "Consumer Defensive", "Exchange": "NASDAQ"},
            {"Symbol": "CSCO", "Name": "Cisco Systems Inc.", "Sector": "Technology", "Exchange": "NASDAQ"},
            {"Symbol": "ADBE", "Name": "Adobe Inc.", "Sector": "Technology", "Exchange": "NASDAQ"},
            {"Symbol": "AMD", "Name": "Advanced Micro Devices Inc.", "Sector": "Technology", "Exchange": "NASDAQ"},
            {"Symbol": "QCOM", "Name": "Qualcomm Inc.", "Sector": "Technology", "Exchange": "NASDAQ"},
            {"Symbol": "COST", "Name": "Costco Wholesale Corporation", "Sector": "Consumer Defensive", "Exchange": "NASDAQ"},
            {"Symbol": "TXN", "Name": "Texas Instruments Inc.", "Sector": "Technology", "Exchange": "NASDAQ"},
            {"Symbol": "TMUS", "Name": "T-Mobile US Inc.", "Sector": "Communication Services", "Exchange": "NASDAQ"},
            {"Symbol": "SBUX", "Name": "Starbucks Corporation", "Sector": "Consumer Cyclical", "Exchange": "NASDAQ"},
        ]
        
        ticker_df = pd.DataFrame(tickers)
        logging.info(f"Created a list of {len(ticker_df)} popular NASDAQ tickers")
        return ticker_df
    except Exception as e:
        logging.error(f"Error fetching NASDAQ tickers: {e}")
        return pd.DataFrame(columns=['Symbol', 'Name', 'Sector', 'Exchange'])

def get_nse_tickers():
    """Get list of NSE (National Stock Exchange of India) tickers"""
    try:
        # First try the NSE official URL
        url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            ticker_df = pd.read_csv(pd.io.common.StringIO(response.text))
            if not ticker_df.empty:
                # Add a column to indicate exchange
                ticker_df['Exchange'] = 'NSE'
                logging.info(f"Successfully fetched {len(ticker_df)} NSE tickers from archives")
                return ticker_df
        
        # If official source failed, create a list of popular NSE tickers
        logging.info("Using alternative method for NSE tickers")
        tickers = [
            {"Symbol": "RELIANCE", "Company Name": "Reliance Industries Ltd.", "Industry": "Energy", "Exchange": "NSE"},
            {"Symbol": "TCS", "Company Name": "Tata Consultancy Services Ltd.", "Industry": "Information Technology", "Exchange": "NSE"},
            {"Symbol": "HDFCBANK", "Company Name": "HDFC Bank Ltd.", "Industry": "Financial Services", "Exchange": "NSE"},
            {"Symbol": "INFY", "Company Name": "Infosys Ltd.", "Industry": "Information Technology", "Exchange": "NSE"},
            {"Symbol": "HINDUNILVR", "Company Name": "Hindustan Unilever Ltd.", "Industry": "Consumer Goods", "Exchange": "NSE"},
            {"Symbol": "HDFC", "Company Name": "Housing Development Finance Corporation Ltd.", "Industry": "Financial Services", "Exchange": "NSE"},
            {"Symbol": "ICICIBANK", "Company Name": "ICICI Bank Ltd.", "Industry": "Financial Services", "Exchange": "NSE"},
            {"Symbol": "KOTAKBANK", "Company Name": "Kotak Mahindra Bank Ltd.", "Industry": "Financial Services", "Exchange": "NSE"},
            {"Symbol": "ITC", "Company Name": "ITC Ltd.", "Industry": "Consumer Goods", "Exchange": "NSE"},
            {"Symbol": "BHARTIARTL", "Company Name": "Bharti Airtel Ltd.", "Industry": "Telecommunications", "Exchange": "NSE"},
            {"Symbol": "SBIN", "Company Name": "State Bank of India", "Industry": "Financial Services", "Exchange": "NSE"},
            {"Symbol": "BAJFINANCE", "Company Name": "Bajaj Finance Ltd.", "Industry": "Financial Services", "Exchange": "NSE"},
            {"Symbol": "ASIANPAINT", "Company Name": "Asian Paints Ltd.", "Industry": "Consumer Goods", "Exchange": "NSE"},
            {"Symbol": "WIPRO", "Company Name": "Wipro Ltd.", "Industry": "Information Technology", "Exchange": "NSE"},
            {"Symbol": "HCLTECH", "Company Name": "HCL Technologies Ltd.", "Industry": "Information Technology", "Exchange": "NSE"},
            {"Symbol": "AXISBANK", "Company Name": "Axis Bank Ltd.", "Industry": "Financial Services", "Exchange": "NSE"},
            {"Symbol": "ULTRACEMCO", "Company Name": "UltraTech Cement Ltd.", "Industry": "Construction Materials", "Exchange": "NSE"},
            {"Symbol": "MARUTI", "Company Name": "Maruti Suzuki India Ltd.", "Industry": "Automobile", "Exchange": "NSE"},
            {"Symbol": "SUNPHARMA", "Company Name": "Sun Pharmaceutical Industries Ltd.", "Industry": "Healthcare", "Exchange": "NSE"},
            {"Symbol": "TATASTEEL", "Company Name": "Tata Steel Ltd.", "Industry": "Metals & Mining", "Exchange": "NSE"}
        ]
        
        # Create DataFrame from the manual list
        ticker_df = pd.DataFrame(tickers)
        logging.info(f"Created a list of {len(ticker_df)} popular NSE tickers")
        return ticker_df
    except Exception as e:
        logging.error(f"Error fetching NSE tickers: {e}")
        return pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Exchange'])


def convert_usd_to_inr(usd_value, use_cache=True):
    """
    Convert USD value to INR using the exchange rate API or cached rate
    """
    if usd_value is None:
        return None
        
    try:
        # Default exchange rate (in case API fails)
        default_rate = 83.5
        
        # Try to get the exchange rate from the API
        if use_cache:
            cache_dir = ensure_cache_dir()
            cache_path = cache_dir / "usd_inr_rate.json"
            
            # Check if we have a recent cached rate (less than 12 hours old)
            if cache_path.exists():
                try:
                    with open(cache_path, 'r') as f:
                        cached_data = json.load(f)
                    timestamp = cached_data.get('timestamp', 0)
                    age_hours = (datetime.now().timestamp() - timestamp) / 3600
                    if age_hours < 12:  # Use cache if less than 12 hours old
                        logging.info(f"Using cached USD to INR rate: {cached_data['rate']}")
                        return usd_value * cached_data['rate']
                except Exception as e:
                    logging.error(f"Error reading exchange rate cache: {e}")
        
        # Try to get the rate from the API
        import os
        import requests
        
        api_key = os.environ.get('EXCHANGE_RATE_API_KEY')
        if api_key:
            try:
                url = f"http://api.exchangeratesapi.io/v1/latest?access_key={api_key}&base=USD&symbols=INR"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if 'rates' in data and 'INR' in data['rates']:
                        rate = data['rates']['INR']
                        
                        # Save to cache
                        if use_cache:
                            try:
                                with open(cache_path, 'w') as f:
                                    json.dump({
                                        'rate': rate,
                                        'timestamp': datetime.now().timestamp()
                                    }, f)
                                logging.info(f"Cached new USD to INR rate: {rate}")
                            except Exception as e:
                                logging.error(f"Error saving exchange rate to cache: {e}")
                        
                        return usd_value * rate
            except Exception as e:
                logging.error(f"Error fetching exchange rate from API: {e}")
        
        # Use the predefined rate if API call fails
        return usd_value * default_rate
    except Exception as e:
        logging.error(f"Error converting USD to INR: {e}")
        return usd_value * 83.5  # Fallback to default rate


def get_bse_tickers():
    """Get a list of popular BSE tickers"""
    try:
        # For now, return a manually curated list of popular BSE tickers
        logging.info("Fetching BSE tickers from manual list")
        
        # Create a list of dictionaries with ticker information
        tickers = [
            {"Symbol": "500325", "Company Name": "Reliance Industries Ltd.", "Industry": "Oil & Gas", "Exchange": "BSE"},
            {"Symbol": "500112", "Company Name": "State Bank of India", "Industry": "Banking", "Exchange": "BSE"},
            {"Symbol": "500010", "Company Name": "Housing Development Finance Corporation Ltd.", "Industry": "Financial Services", "Exchange": "BSE"},
            {"Symbol": "500180", "Company Name": "HDFC Bank Ltd.", "Industry": "Banking", "Exchange": "BSE"},
            {"Symbol": "500209", "Company Name": "Infosys Ltd.", "Industry": "Information Technology", "Exchange": "BSE"},
            {"Symbol": "500570", "Company Name": "Tata Motors Ltd.", "Industry": "Automobile", "Exchange": "BSE"},
            {"Symbol": "500875", "Company Name": "ITC Ltd.", "Industry": "FMCG", "Exchange": "BSE"},
            {"Symbol": "500696", "Company Name": "Hindustan Unilever Ltd.", "Industry": "FMCG", "Exchange": "BSE"},
            {"Symbol": "500520", "Company Name": "Mahindra & Mahindra Ltd.", "Industry": "Automobile", "Exchange": "BSE"},
            {"Symbol": "532174", "Company Name": "ICICI Bank Ltd.", "Industry": "Banking", "Exchange": "BSE"}
        ]
        
        # Create DataFrame from the manual list
        ticker_df = pd.DataFrame(tickers)
        logging.info(f"Created a list of {len(ticker_df)} popular BSE tickers")
        return ticker_df
    except Exception as e:
        logging.error(f"Error fetching BSE tickers: {e}")
        return pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Exchange'])
