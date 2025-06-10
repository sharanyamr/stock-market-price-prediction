// Sentiment Comments JavaScript
document.addEventListener('DOMContentLoaded', function() {
    console.log('Sentiment comments JS loaded');
    
    // Get DOM elements
    const commentForm = document.getElementById('commentForm');
    const commentInput = document.getElementById('commentInput');
    const symbolInput = document.getElementById('symbolInput');
    const postCommentBtn = document.getElementById('postCommentBtn');
    const commentsList = document.getElementById('commentsList');
    
    // Function to show alerts
    function showAlert(message, type) {
        const alertContainer = document.getElementById('alertContainer');
        if (!alertContainer) return;
        
        const alertElement = document.createElement('div');
        alertElement.className = `alert alert-${type} alert-dismissible fade show`;
        alertElement.setAttribute('role', 'alert');
        
        alertElement.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        alertContainer.appendChild(alertElement);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            alertElement.classList.remove('show');
            setTimeout(() => alertElement.remove(), 300);
        }, 5000);
    }
    
    // Function to add a comment to the DOM
    function addCommentToDOM(comment) {
        const commentElement = document.createElement('div');
        commentElement.className = 'comment-container mb-4';
        commentElement.dataset.commentId = comment.id;
        
        // Generate replies HTML
        let repliesHTML = '';
        if (comment.replies && comment.replies.length > 0) {
            comment.replies.forEach(reply => {
                repliesHTML += `
                    <div class="reply mb-2">
                        <div class="reply-header d-flex justify-content-between">
                            <span class="reply-username"><i class="fas fa-user-circle me-1"></i> ${reply.username}</span>
                            <span class="reply-timestamp text-muted"><i class="fas fa-clock me-1"></i> ${reply.created_at}</span>
                        </div>
                        <div class="reply-body my-2">
                            ${reply.reply}
                        </div>
                    </div>
                `;
            });
        }
        
        commentElement.innerHTML = `
            <div class="comment">
                <div class="comment-header d-flex justify-content-between">
                    <span class="comment-username"><i class="fas fa-user-circle me-1"></i> ${comment.username}</span>
                    <span class="comment-timestamp text-muted"><i class="fas fa-clock me-1"></i> ${comment.created_at}</span>
                </div>
                <div class="comment-body my-2">
                    ${comment.comment}
                </div>
                <div class="comment-footer">
                    <span class="badge bg-${comment.sentiment === 'positive' ? 'success' : (comment.sentiment === 'negative' ? 'danger' : 'warning')}">
                        ${comment.sentiment.toUpperCase()}
                    </span>
                    <button class="btn btn-sm btn-outline-primary ms-2 reply-button">
                        <i class="fas fa-reply me-1"></i> Reply
                    </button>
                </div>
                <div class="reply-form-container mt-3" style="display: none;">
                    <div class="input-group">
                        <input type="text" class="form-control reply-input" placeholder="Write a reply...">
                        <button class="btn btn-primary submit-reply-btn" data-comment-id="${comment.id}">Reply</button>
                    </div>
                </div>
            </div>
            <div class="comment-replies ms-4 mt-2">
                ${repliesHTML}
            </div>
        `;
        
        // Clear current "no comments" message if present
        if (commentsList.querySelector('.text-center.py-5')) {
            commentsList.innerHTML = '';
        }
        
        commentsList.prepend(commentElement);
        
        // Set up reply button functionality
        const replyBtn = commentElement.querySelector('.reply-button');
        if (replyBtn) {
            const replyForm = commentElement.querySelector('.reply-form-container');
            
            replyBtn.addEventListener('click', function() {
                replyForm.style.display = replyForm.style.display === 'none' ? 'block' : 'none';
            });
        }
        
        // Set up reply submission
        const submitReplyBtn = commentElement.querySelector('.submit-reply-btn');
        if (submitReplyBtn) {
            const replyInput = commentElement.querySelector('.reply-input');
            const repliesContainer = commentElement.querySelector('.comment-replies');
            const commentId = submitReplyBtn.dataset.commentId;
            
            submitReplyBtn.addEventListener('click', function() {
                const replyText = replyInput.value.trim();
                if (!replyText) return;
                
                // Send reply to the API
                fetch('/api/sentiment/replies', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        comment_id: commentId,
                        reply: replyText
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // Add reply to the DOM
                        const replyElement = document.createElement('div');
                        replyElement.className = 'reply mb-2';
                        replyElement.innerHTML = `
                            <div class="reply-header d-flex justify-content-between">
                                <span class="reply-username"><i class="fas fa-user-circle me-1"></i> ${data.reply.username}</span>
                                <span class="reply-timestamp text-muted"><i class="fas fa-clock me-1"></i> ${data.reply.created_at}</span>
                            </div>
                            <div class="reply-body my-2">
                                ${data.reply.reply}
                            </div>
                        `;
                        
                        repliesContainer.appendChild(replyElement);
                        replyInput.value = '';
                        replyForm.style.display = 'none';
                        
                        showAlert('Reply posted successfully', 'success');
                    } else {
                        showAlert(data.error || 'Error posting reply', 'danger');
                    }
                })
                .catch(error => {
                    console.error('Error posting reply:', error);
                    showAlert('Error posting reply. Please try again.', 'danger');
                });
            });
        }
    }
    
    // Handle direct form submission
    if (commentForm) {
        commentForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get form data
            const symbol = symbolInput.value;
            const comment = commentInput.value.trim();
            const sentimentRadio = document.querySelector('input[name="sentimentRadio"]:checked');
            const sentiment = sentimentRadio ? sentimentRadio.value : 'neutral';
            
            if (!comment) {
                showAlert('Please enter a comment', 'warning');
                return;
            }
            
            console.log('Submitting comment:', { symbol, comment, sentiment });
            
            // Disable the button while submitting
            if (postCommentBtn) {
                postCommentBtn.disabled = true;
                postCommentBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Posting...';
            }
            
            // Send the comment to the server using XMLHttpRequest instead of fetch
            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/api/sentiment/comments', true);
            xhr.setRequestHeader('Content-Type', 'application/json');
            
            xhr.onload = function() {
                // Re-enable the button
                if (postCommentBtn) {
                    postCommentBtn.disabled = false;
                    postCommentBtn.innerHTML = 'Post Comment';
                }
                
                if (xhr.status === 200) {
                    try {
                        const data = JSON.parse(xhr.responseText);
                        console.log('Response data:', data);
                        
                        if (data.success) {
                            // Add the new comment to the DOM
                            const newComment = {
                                id: data.comment.id,
                                username: data.comment.username,
                                comment: data.comment.comment,
                                sentiment: data.comment.sentiment,
                                created_at: data.comment.created_at,
                                replies: []
                            };
                            
                            addCommentToDOM(newComment);
                            commentInput.value = '';
                            
                            showAlert('Comment posted successfully', 'success');
                        } else {
                            showAlert(data.error || 'Error posting comment', 'danger');
                        }
                    } catch (e) {
                        console.error('Error parsing response:', e);
                        showAlert('Error processing server response', 'danger');
                    }
                } else {
                    console.error('Server returned status:', xhr.status);
                    showAlert('Error posting comment. Server returned status: ' + xhr.status, 'danger');
                }
            };
            
            xhr.onerror = function() {
                console.error('Request error');
                if (postCommentBtn) {
                    postCommentBtn.disabled = false;
                    postCommentBtn.innerHTML = 'Post Comment';
                }
                showAlert('Network error. Please try again.', 'danger');
            };
            
            xhr.send(JSON.stringify({
                symbol: symbol,
                comment: comment,
                sentiment: sentiment
            }));
        });
    }
    
    // Initialize reply buttons for existing comments
    document.querySelectorAll('.reply-button').forEach(button => {
        button.addEventListener('click', function() {
            const replyForm = this.closest('.comment').querySelector('.reply-form-container');
            replyForm.style.display = replyForm.style.display === 'none' ? 'block' : 'none';
        });
    });
    
    // Initialize reply submission for existing comments
    document.querySelectorAll('.submit-reply-btn').forEach(button => {
        button.addEventListener('click', function() {
            const commentId = this.dataset.commentId;
            const replyInput = this.closest('.input-group').querySelector('.reply-input');
            const replyText = replyInput.value.trim();
            const repliesContainer = this.closest('.comment-container').querySelector('.comment-replies');
            
            if (!replyText) return;
            
            // Send reply to the API
            fetch('/api/sentiment/replies', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    comment_id: commentId,
                    reply: replyText
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Add reply to the DOM
                    const replyElement = document.createElement('div');
                    replyElement.className = 'reply mb-2';
                    replyElement.innerHTML = `
                        <div class="reply-header d-flex justify-content-between">
                            <span class="reply-username"><i class="fas fa-user-circle me-1"></i> ${data.reply.username}</span>
                            <span class="reply-timestamp text-muted"><i class="fas fa-clock me-1"></i> ${data.reply.created_at}</span>
                        </div>
                        <div class="reply-body my-2">
                            ${data.reply.reply}
                        </div>
                    `;
                    
                    repliesContainer.appendChild(replyElement);
                    replyInput.value = '';
                    this.closest('.reply-form-container').style.display = 'none';
                    
                    showAlert('Reply posted successfully', 'success');
                } else {
                    showAlert(data.error || 'Error posting reply', 'danger');
                }
            })
            .catch(error => {
                console.error('Error posting reply:', error);
                showAlert('Error posting reply. Please try again.', 'danger');
            });
        });
    });
});
