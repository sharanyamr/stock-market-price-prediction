import os
import logging
import re
from datetime import datetime

from flask import Flask
from markupsafe import Markup
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_login import LoginManager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
print(f"[DEBUG] DATABASE_URL loaded: {os.environ.get('DATABASE_URL')}")  # Debug print

# Configure logging - reduce to INFO level for production
logging.basicConfig(level=logging.INFO)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key_for_development")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  # needed for url_for to generate with https

# Configure the database with optimized connection pooling
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_size": 10,  # Increase connection pool size
    "max_overflow": 20,  # Allow more connections when pool is full
    "pool_recycle": 300,  # Recycle connections after 5 minutes
    "pool_pre_ping": True,  # Test connections before using them
    "pool_timeout": 30,  # Timeout for getting connection from pool
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Import db from extensions
from extensions import db

# Initialize Flask extensions
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    from models import User
    return User.query.get(int(user_id))

# Add custom Jinja2 filters
@app.template_filter('slice')
def slice_filter(iterable, start, end=None):
    """Custom filter for slicing in Jinja2 templates"""
    if end:
        return list(iterable)[start:end]
    else:
        return list(iterable)[start:]

# Context processors
@app.context_processor
def inject_year():
    return {'current_year': datetime.now().year}

# Custom Jinja filters
@app.template_filter('nl2br')
def nl2br_filter(text):
    if text:
        return Markup(text.replace('\n', '<br>'))
    return ""

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return "Page not found", 404

@app.errorhandler(500)
def internal_server_error(e):
    return "Internal server error", 500

# Import models to ensure tables are created
import models

# Create all tables in the app context
with app.app_context():
    db.create_all()

# Import and register blueprints
from routes import main_bp
from auth import auth_bp
from admin import admin_bp
from portfolio import portfolio_bp

app.register_blueprint(main_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(portfolio_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
