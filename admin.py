from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from datetime import datetime
import logging

from extensions import db
from models import User, EducationArticle, NewsArticle, SentimentComment

# Initialize blueprint
admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.before_request
def check_admin():
    """Check if the current user is an admin"""
    if not current_user.is_authenticated or not current_user.is_admin:
        flash('You do not have permission to access this page', 'danger')
        return redirect(url_for('main.index'))

@admin_bp.route('/')
@login_required
def dashboard():
    """Admin dashboard"""
    # Get counts for dashboard stats
    try:
        user_count = User.query.count()
        article_count = EducationArticle.query.count()
        news_count = NewsArticle.query.count()
        comment_count = SentimentComment.query.count()
        
        # Get recent users
        recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
        
        # Get recent comments
        recent_comments = SentimentComment.query.order_by(SentimentComment.created_at.desc()).limit(5).all()
    except Exception as e:
        logging.error(f"Admin dashboard error: {e}")
        user_count = article_count = news_count = comment_count = 0
        recent_users = recent_comments = []
        flash('Error loading admin dashboard data', 'danger')
    
    return render_template('admin/dashboard.html', 
                           user_count=user_count,
                           article_count=article_count,
                           news_count=news_count,
                           comment_count=comment_count,
                           recent_users=recent_users,
                           recent_comments=recent_comments)

@admin_bp.route('/users')
@login_required
def users():
    """Manage users"""
    try:
        users_list = User.query.order_by(User.username).all()
    except Exception as e:
        logging.error(f"Admin users list error: {e}")
        users_list = []
        flash('Error loading users', 'danger')
    
    return render_template('admin/users.html', users=users_list)

@admin_bp.route('/user/<int:user_id>/toggle-admin', methods=['POST'])
@login_required
def toggle_admin(user_id):
    """Toggle admin status for a user"""
    try:
        user = User.query.get_or_404(user_id)
        
        # Prevent removing own admin privileges
        if user.id == current_user.id:
            flash('You cannot remove your own admin privileges', 'danger')
            return redirect(url_for('admin.users'))
        
        user.is_admin = not user.is_admin
        db.session.commit()
        
        flash(f'Admin status for {user.username} has been {"granted" if user.is_admin else "revoked"}', 'success')
    except Exception as e:
        db.session.rollback()
        logging.error(f"Toggle admin error: {e}")
        flash('Error updating user admin status', 'danger')
    
    return redirect(url_for('admin.users'))

@admin_bp.route('/articles')
@login_required
def articles():
    """Manage educational articles"""
    try:
        articles_list = EducationArticle.query.order_by(EducationArticle.created_at.desc()).all()
    except Exception as e:
        logging.error(f"Admin articles list error: {e}")
        articles_list = []
        flash('Error loading articles', 'danger')
    
    return render_template('admin/articles.html', articles=articles_list)

@admin_bp.route('/article/new', methods=['GET', 'POST'])
@login_required
def new_article():
    """Create a new educational article"""
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        category = request.form.get('category')
        featured = 'featured' in request.form
        
        if not title or not content:
            flash('Title and content are required', 'danger')
            return render_template('admin/article_form.html')
        
        try:
            article = EducationArticle(
                title=title,
                content=content,
                author=f"{current_user.first_name} {current_user.last_name}",
                category=category,
                featured=featured
            )
            
            db.session.add(article)
            db.session.commit()
            
            flash('Article created successfully', 'success')
            return redirect(url_for('admin.articles'))
        except Exception as e:
            db.session.rollback()
            logging.error(f"Create article error: {e}")
            flash('Error creating article', 'danger')
    
    return render_template('admin/article_form.html')

@admin_bp.route('/article/<int:article_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_article(article_id):
    """Edit an educational article"""
    try:
        article = EducationArticle.query.get_or_404(article_id)
    except Exception as e:
        logging.error(f"Edit article error: {e}")
        flash('Article not found', 'danger')
        return redirect(url_for('admin.articles'))
    
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        category = request.form.get('category')
        featured = 'featured' in request.form
        
        if not title or not content:
            flash('Title and content are required', 'danger')
            return render_template('admin/article_form.html', article=article)
        
        try:
            article.title = title
            article.content = content
            article.category = category
            article.featured = featured
            article.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            flash('Article updated successfully', 'success')
            return redirect(url_for('admin.articles'))
        except Exception as e:
            db.session.rollback()
            logging.error(f"Update article error: {e}")
            flash('Error updating article', 'danger')
    
    return render_template('admin/article_form.html', article=article)

@admin_bp.route('/article/<int:article_id>/delete', methods=['POST'])
@login_required
def delete_article(article_id):
    """Delete an educational article"""
    try:
        article = EducationArticle.query.get_or_404(article_id)
        
        db.session.delete(article)
        db.session.commit()
        
        flash('Article deleted successfully', 'success')
    except Exception as e:
        db.session.rollback()
        logging.error(f"Delete article error: {e}")
        flash('Error deleting article', 'danger')
    
    return redirect(url_for('admin.articles'))

@admin_bp.route('/comments')
@login_required
def comments():
    """Manage sentiment comments"""
    try:
        comments_list = SentimentComment.query.order_by(SentimentComment.created_at.desc()).all()
    except Exception as e:
        logging.error(f"Admin comments list error: {e}")
        comments_list = []
        flash('Error loading comments', 'danger')
    
    return render_template('admin/comments.html', comments=comments_list)

@admin_bp.route('/comment/<int:comment_id>/delete', methods=['POST'])
@login_required
def delete_comment(comment_id):
    """Delete a sentiment comment"""
    try:
        comment = SentimentComment.query.get_or_404(comment_id)
        
        # Delete associated replies
        for reply in comment.replies:
            db.session.delete(reply)
        
        db.session.delete(comment)
        db.session.commit()
        
        flash('Comment and associated replies deleted successfully', 'success')
    except Exception as e:
        db.session.rollback()
        logging.error(f"Delete comment error: {e}")
        flash('Error deleting comment', 'danger')
    
    return redirect(url_for('admin.comments'))

@admin_bp.route('/news')
@login_required
def news():
    """Manage news articles"""
    try:
        news_list = NewsArticle.query.order_by(NewsArticle.published_at.desc()).all()
    except Exception as e:
        logging.error(f"Admin news list error: {e}")
        news_list = []
        flash('Error loading news articles', 'danger')
    
    return render_template('admin/news.html', news=news_list)

@admin_bp.route('/news/refresh', methods=['POST'])
@login_required
def refresh_news():
    """Refresh news articles from API"""
    from utils import fetch_latest_news
    
    try:
        count = fetch_latest_news()
        flash(f'Successfully fetched {count} news articles', 'success')
    except Exception as e:
        logging.error(f"Refresh news error: {e}")
        flash('Error refreshing news articles', 'danger')
    
    return redirect(url_for('admin.news'))
