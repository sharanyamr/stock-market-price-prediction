from app import app  # noqa: F401
import os

if __name__ == "__main__":
    # Use threaded mode for better performance
    # Set debug to False in production for better performance
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    app.run(
        host="0.0.0.0", 
        port=5000, 
        debug=debug_mode,
        threaded=True,
        # Increase request handling capacity
        processes=1  # Use 1 process with threading for better performance
    )
