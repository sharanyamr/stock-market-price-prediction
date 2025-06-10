from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import logging

from extensions import db
from models import User

# Initialize blueprint
auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    # If user is already logged in, redirect to dashboard
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = 'remember' in request.form
        
        # Validate inputs
        if not username or not password:
            flash('Please enter both username and password', 'danger')
            return render_template('login.html')
        
        # Check if user exists
        try:
            user = User.query.filter_by(username=username).first()
            
            if user and user.check_password(password):
                login_user(user, remember=remember)
                user.last_login = datetime.utcnow()
                db.session.commit()
                
                # Get next page or default to dashboard
                next_page = request.args.get('next')
                if not next_page or not next_page.startswith('/'):
                    next_page = url_for('main.dashboard')
                
                flash('Login successful!', 'success')
                return redirect(next_page)
            else:
                flash('Invalid username or password', 'danger')
        except Exception as e:
            logging.error(f"Login error: {e}")
            flash('An error occurred during login. Please try again.', 'danger')
    
    return render_template('login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page"""
    # If user is already logged in, redirect to dashboard
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        
        # Validate inputs
        if not username or not email or not password:
            flash('Please fill in all required fields', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
        
        # Check if username or email already exists
        try:
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                flash('Username already exists', 'danger')
                return render_template('register.html')
            
            existing_email = User.query.filter_by(email=email).first()
            if existing_email:
                flash('Email already registered', 'danger')
                return render_template('register.html')
            
            # Create new user
            new_user = User(
                username=username,
                email=email,
                first_name=first_name,
                last_name=last_name
            )
            new_user.set_password(password)
            
            db.session.add(new_user)
            db.session.commit()
            
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('auth.login'))
        except Exception as e:
            db.session.rollback()
            logging.error(f"Registration error: {e}")
            flash('An error occurred during registration. Please try again.', 'danger')
    
    return render_template('register.html')

@auth_bp.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('main.index'))

@auth_bp.route('/profile')
@login_required
def profile():
    """User profile page"""
    return render_template('profile.html')

@auth_bp.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    """Edit user profile"""
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        
        try:
            # Check if email changed and already exists
            if email != current_user.email:
                existing_email = User.query.filter_by(email=email).first()
                if existing_email:
                    flash('Email already registered', 'danger')
                    return redirect(url_for('auth.edit_profile'))
            
            # Update user data
            current_user.first_name = first_name
            current_user.last_name = last_name
            current_user.email = email
            
            db.session.commit()
            flash('Profile updated successfully', 'success')
            return redirect(url_for('auth.profile'))
        except Exception as e:
            db.session.rollback()
            logging.error(f"Profile update error: {e}")
            flash('An error occurred while updating your profile. Please try again.', 'danger')
    
    return render_template('edit_profile.html')

@auth_bp.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Change user password"""
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate inputs
        if not current_password or not new_password or not confirm_password:
            flash('Please fill in all fields', 'danger')
            return render_template('change_password.html')
        
        if new_password != confirm_password:
            flash('New passwords do not match', 'danger')
            return render_template('change_password.html')
        
        # Check current password
        if not current_user.check_password(current_password):
            flash('Current password is incorrect', 'danger')
            return render_template('change_password.html')
        
        try:
            # Update password
            current_user.set_password(new_password)
            db.session.commit()
            
            flash('Password changed successfully', 'success')
            return redirect(url_for('auth.profile'))
        except Exception as e:
            db.session.rollback()
            logging.error(f"Password change error: {e}")
            flash('An error occurred while changing your password. Please try again.', 'danger')
    
    return render_template('change_password.html')
