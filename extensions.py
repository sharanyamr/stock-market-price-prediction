"""
Extensions module to avoid circular imports
"""
from flask_sqlalchemy import SQLAlchemy

# Create extensions without initializing them
db = SQLAlchemy()
