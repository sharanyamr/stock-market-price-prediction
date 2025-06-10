import logging
import random
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
import yfinance as yf
import pandas as pd
import requests

from app import db
from models import SentimentComment, SentimentReply, NewsArticle
from utils import ensure_cache_dir

# Initialize sentiment blueprint
sentiment_bp = Blueprint('sentiment', __name__, url_prefix='/sentiment')

# Ensure NLTK resources are downloaded
def ensure_nltk_resources():
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

# Initialize NLTK resources
ensure_nltk_resources()

# Cache for sentiment data
def get_sentiment_cache_path(symbol):
    cache_dir = ensure_cache_dir()
    return cache_dir / f"sentiment_{symbol}.json"

def save_sentiment_to_cache(data, symbol):
    cache_path = get_sentiment_cache_path(symbol)
    try:
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        logging.info(f"Cached sentiment data for {symbol}")
    except Exception as e:
        logging.error(f"Error caching sentiment data: {e}")

def load_sentiment_from_cache(symbol, max_age_hours=6):
    cache_path = get_sentiment_cache_path(symbol)
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cached_data = json.load(f)
        
        # Check if cache is too old
        timestamp = cached_data.get('timestamp', 0)
        age_hours = (datetime.now().timestamp() - timestamp) / 3600
        if age_hours > max_age_hours:
            logging.info(f"Sentiment cache for {symbol} is {age_hours:.1f} hours old, refreshing")
            return None
        
        logging.info(f"Using cached sentiment data for {symbol}")
        return cached_data
    except Exception as e:
        logging.error(f"Error loading sentiment cache: {e}")
    
    return None

# Function to get sentiment analysis for a stock
def get_sentiment_analysis(symbol):
    """Get sentiment analysis for a stock symbol
    
    Args:
        symbol (str): Stock ticker symbol
        
    Returns:
        dict: Sentiment analysis data
    """
    try:
        # Check cache first
        cached_data = load_sentiment_from_cache(symbol)
        if cached_data:
            return cached_data
        
        # Initialize the sentiment analyzer
        sid = SentimentIntensityAnalyzer()
        
        # Clean symbol for search (remove exchange suffix)
        search_symbol = symbol.split('.')[0] if '.' in symbol else symbol
        
        # 1. Get sentiment from news articles
        news_sentiments = []
        
        # Search for articles mentioning the stock
        recent_articles = NewsArticle.query.filter(
            (NewsArticle.title.contains(search_symbol)) | 
            (NewsArticle.content.contains(search_symbol))
        ).order_by(NewsArticle.published_at.desc()).limit(10).all()
        
        # If no articles found, try to get some general market news
        if not recent_articles:
            recent_articles = NewsArticle.query.filter(
                (NewsArticle.category == 'markets') | 
                (NewsArticle.category == 'business')
            ).order_by(NewsArticle.published_at.desc()).limit(5).all()
        
        # Analyze sentiment of each article
        for article in recent_articles:
            # Combine title and content for analysis
            text = f"{article.title}. {article.content[:500]}"
            sentiment_scores = sid.polarity_scores(text)
            news_sentiments.append({
                'source': 'news',
                'text': article.title,
                'url': article.url,
                'scores': sentiment_scores,
                'date': article.published_at.strftime('%Y-%m-%d') if article.published_at else 'Unknown'
            })
        
        # 2. Get sentiment from user comments
        comment_sentiments = []
        
        # Get recent comments for this stock
        recent_comments = SentimentComment.query.filter_by(
            stock_symbol=symbol
        ).order_by(SentimentComment.created_at.desc()).limit(10).all()
        
        # Analyze sentiment of each comment
        for comment in recent_comments:
            sentiment_scores = sid.polarity_scores(comment.comment)
            comment_sentiments.append({
                'source': 'user',
                'text': comment.comment,
                'scores': sentiment_scores,
                'date': comment.created_at.strftime('%Y-%m-%d'),
                'user_id': comment.user_id,
                'comment_id': comment.id
            })
        
        # 3. Generate mock social media sentiment (in a real app, you'd use Twitter/Reddit API)
        social_sentiments = []
        
        # This simulates what you'd get from a social media API
        mock_social_media = [
            f"Really bullish on {symbol} right now! The company's latest product launch looks promising.",
            f"Not sure about {symbol}, their last earnings report was concerning.",
            f"Just bought more {symbol} shares. Long-term hold for me.",
            f"{symbol} is overvalued at current prices. Waiting for a pullback.",
            f"The management team at {symbol} is making smart strategic decisions.",
            f"Technical indicators for {symbol} look positive, expecting a breakout soon.",
            f"Concerned about {symbol}'s exposure to current market conditions.",
            f"Dividend yield for {symbol} makes it an attractive income stock."
        ]
        
        # Randomly select 5 mock posts and analyze sentiment
        selected_posts = random.sample(mock_social_media, min(5, len(mock_social_media)))
        
        for post in selected_posts:
            sentiment_scores = sid.polarity_scores(post)
            social_sentiments.append({
                'source': 'social',
                'text': post,
                'scores': sentiment_scores,
                'date': (datetime.now() - timedelta(days=random.randint(0, 5))).strftime('%Y-%m-%d')
            })
        
        # Combine all sentiment sources
        all_sentiments = news_sentiments + comment_sentiments + social_sentiments
        
        # Calculate overall sentiment metrics
        positive_count = sum(1 for item in all_sentiments if item['scores']['compound'] > 0.05)
        negative_count = sum(1 for item in all_sentiments if item['scores']['compound'] < -0.05)
        neutral_count = len(all_sentiments) - positive_count - negative_count
        
        # Calculate average compound score
        if all_sentiments:
            avg_compound = sum(item['scores']['compound'] for item in all_sentiments) / len(all_sentiments)
        else:
            avg_compound = 0
        
        # Determine overall sentiment
        if avg_compound > 0.05:
            overall_sentiment = 'positive'
        elif avg_compound < -0.05:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        # Create sentiment result
        sentiment_result = {
            'symbol': symbol,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'overall_sentiment': overall_sentiment,
            'avg_compound': round(avg_compound, 2),
            'news_sentiments': news_sentiments,
            'user_sentiments': comment_sentiments,
            'social_sentiments': social_sentiments,
            'timestamp': datetime.now().timestamp()
        }
        
        # Cache the result
        save_sentiment_to_cache(sentiment_result, symbol)
        
        return sentiment_result
        
    except Exception as e:
        logging.error(f"Error getting sentiment analysis: {e}")
        # Return a default sentiment if there's an error
        return {
            'symbol': symbol,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'overall_sentiment': 'neutral',
            'avg_compound': 0.0,
            'news_sentiments': [],
            'user_sentiments': [],
            'social_sentiments': [],
            'error': str(e)
        }

@sentiment_bp.route('/')
@login_required
def index():
    """Sentiment analysis page"""
    symbol = request.args.get('symbol', 'AAPL')
    
    try:
        # Get sentiment analysis for the selected stock
        sentiment_data = get_sentiment_analysis(symbol)
        
        # Get user comments for this stock
        comments = SentimentComment.query.filter_by(
            stock_symbol=symbol
        ).order_by(SentimentComment.created_at.desc()).all()
        
        # Format comments with user info and replies
        formatted_comments = []
        for comment in comments:
            # Get user info
            user = comment.user
            
            # Get replies
            replies = SentimentReply.query.filter_by(
                comment_id=comment.id
            ).order_by(SentimentReply.created_at).all()
            
            formatted_replies = []
            for reply in replies:
                reply_user = reply.user
                formatted_replies.append({
                    'id': reply.id,
                    'user_id': reply.user_id,
                    'username': reply_user.username,
                    'reply': reply.reply,
                    'created_at': reply.created_at.strftime('%Y-%m-%d %H:%M')
                })
            
            formatted_comments.append({
                'id': comment.id,
                'user_id': comment.user_id,
                'username': user.username,
                'comment': comment.comment,
                'sentiment': comment.sentiment,
                'created_at': comment.created_at.strftime('%Y-%m-%d %H:%M'),
                'replies': formatted_replies
            })
        
    except Exception as e:
        logging.error(f"Error in sentiment route: {e}")
        sentiment_data = {
            'symbol': symbol,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'overall_sentiment': 'neutral',
            'avg_compound': 0.0,
            'news_sentiments': [],
            'user_sentiments': [],
            'social_sentiments': [],
            'error': str(e)
        }
        formatted_comments = []
        flash(f"Error fetching sentiment data: {e}", "danger")
    
    return render_template('sentiment.html', 
                           symbol=symbol, 
                           sentiment_data=sentiment_data,
                           comments=formatted_comments)

@sentiment_bp.route('/add_comment', methods=['POST'])
@login_required
def add_comment():
    """Add a comment for sentiment analysis"""
    symbol = request.form.get('symbol')
    comment_text = request.form.get('comment')
    
    if not symbol or not comment_text:
        flash("Symbol and comment are required", "danger")
        return redirect(url_for('sentiment.index', symbol=symbol))
    
    try:
        # Analyze sentiment
        sid = SentimentIntensityAnalyzer()
        sentiment_scores = sid.polarity_scores(comment_text)
        
        # Determine sentiment category
        if sentiment_scores['compound'] > 0.05:
            sentiment = 'positive'
        elif sentiment_scores['compound'] < -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Create comment
        comment = SentimentComment(
            user_id=current_user.id,
            stock_symbol=symbol,
            comment=comment_text,
            sentiment=sentiment,
            created_at=datetime.utcnow()
        )
        
        db.session.add(comment)
        db.session.commit()
        
        flash("Comment added successfully", "success")
        
        # Clear sentiment cache for this symbol
        cache_path = get_sentiment_cache_path(symbol)
        if cache_path.exists():
            cache_path.unlink()
            
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error adding comment: {e}")
        flash(f"Error adding comment: {e}", "danger")
    
    return redirect(url_for('sentiment.index', symbol=symbol))

@sentiment_bp.route('/add_reply', methods=['POST'])
@login_required
def add_reply():
    """Add a reply to a comment"""
    comment_id = request.form.get('comment_id')
    reply_text = request.form.get('reply')
    
    if not comment_id or not reply_text:
        flash("Comment ID and reply are required", "danger")
        return redirect(request.referrer or url_for('sentiment.index'))
    
    try:
        # Get the comment
        comment = SentimentComment.query.get(comment_id)
        if not comment:
            flash("Comment not found", "danger")
            return redirect(request.referrer or url_for('sentiment.index'))
        
        # Create reply
        reply = SentimentReply(
            comment_id=comment_id,
            user_id=current_user.id,
            reply=reply_text,
            created_at=datetime.utcnow()
        )
        
        db.session.add(reply)
        db.session.commit()
        
        flash("Reply added successfully", "success")
        
        # Clear sentiment cache for this symbol
        cache_path = get_sentiment_cache_path(comment.stock_symbol)
        if cache_path.exists():
            cache_path.unlink()
            
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error adding reply: {e}")
        flash(f"Error adding reply: {e}", "danger")
    
    return redirect(request.referrer or url_for('sentiment.index'))

@sentiment_bp.route('/api/sentiment/<symbol>')
def api_sentiment(symbol):
    """API endpoint to get sentiment data for a symbol"""
    try:
        sentiment_data = get_sentiment_analysis(symbol)
        return jsonify(sentiment_data)
    except Exception as e:
        logging.error(f"Error in sentiment API: {e}")
        return jsonify({'error': str(e)}), 500
