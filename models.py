from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

# Import db from a separate module to avoid circular imports
from extensions import db

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(64))
    last_name = db.Column(db.String(64))
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    portfolio_items = db.relationship('Portfolio', backref='user', lazy='dynamic')
    sentiment_comments = db.relationship('SentimentComment', backref='user', lazy='dynamic')
    sentiment_replies = db.relationship('SentimentReply', backref='user', lazy='dynamic')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Portfolio(db.Model):
    __tablename__ = 'portfolio'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    stock_symbol = db.Column(db.String(10), nullable=False)
    quantity = db.Column(db.Float, nullable=False)
    purchase_price = db.Column(db.Float, nullable=False)
    purchase_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Portfolio {self.stock_symbol}: {self.quantity} shares>'

class SentimentComment(db.Model):
    __tablename__ = 'sentiment_comments'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    stock_symbol = db.Column(db.String(10), nullable=False)
    comment = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(10))  # positive, negative, neutral
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    replies = db.relationship('SentimentReply', backref='comment', lazy='dynamic')
    
    def __repr__(self):
        return f'<SentimentComment {self.stock_symbol}: {self.sentiment}>'

class SentimentReply(db.Model):
    __tablename__ = 'sentiment_replies'
    
    id = db.Column(db.Integer, primary_key=True)
    comment_id = db.Column(db.Integer, db.ForeignKey('sentiment_comments.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    reply = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<SentimentReply to comment {self.comment_id}>'

class EducationArticle(db.Model):
    __tablename__ = 'education_articles'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    author = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    category = db.Column(db.String(50))
    featured = db.Column(db.Boolean, default=False)
    
    def __repr__(self):
        return f'<EducationArticle {self.title}>'

class NewsArticle(db.Model):
    __tablename__ = 'news_articles'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    source = db.Column(db.String(100), nullable=False)
    url = db.Column(db.String(500), nullable=False)
    published_at = db.Column(db.DateTime, nullable=False)
    fetched_at = db.Column(db.DateTime, default=datetime.utcnow)
    category = db.Column(db.String(50))
    
    def __repr__(self):
        return f'<NewsArticle {self.title}>'

class PredictionModel(db.Model):
    __tablename__ = 'prediction_models'
    
    id = db.Column(db.Integer, primary_key=True)
    stock_symbol = db.Column(db.String(10), nullable=False)
    model_type = db.Column(db.String(20), nullable=False)  # ARIMA, LSTM, Linear Regression
    prediction_date = db.Column(db.Date, nullable=False)
    predicted_price = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<PredictionModel {self.stock_symbol} {self.model_type}: {self.predicted_price}>'

class SentimentAnalysis(db.Model):
    __tablename__ = 'sentiment_analysis'
    
    id = db.Column(db.Integer, primary_key=True)
    stock_symbol = db.Column(db.String(10), nullable=False)
    date = db.Column(db.Date, nullable=False)
    positive_count = db.Column(db.Integer, default=0)
    negative_count = db.Column(db.Integer, default=0)
    neutral_count = db.Column(db.Integer, default=0)
    overall_sentiment = db.Column(db.String(10))  # positive, negative, neutral
    tweet_examples = db.Column(db.Text)  # JSON string of example tweets
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<SentimentAnalysis {self.stock_symbol}: {self.overall_sentiment}>'

class StockTransaction(db.Model):
    __tablename__ = 'stock_transactions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    stock_symbol = db.Column(db.String(10), nullable=False)
    transaction_type = db.Column(db.String(4), nullable=False)  # buy or sell
    quantity = db.Column(db.Float, nullable=False)
    price = db.Column(db.Float, nullable=False)
    transaction_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<StockTransaction {self.transaction_type} {self.quantity} {self.stock_symbol}>'
