"""
Script to run the FastAPI server for the mine safety injury rate prediction API.
"""
import os
import logging
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def main():
    """Run the FastAPI server."""
    logger.info("Starting Mine Safety Injury Rate Prediction API server")
    
    # Set environment variables if needed
    os.environ['ENVIRONMENT'] = 'development'
    
    # Get the directory of this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set model and data directories
    os.environ['MODELS_DIR'] = os.path.join(base_dir, 'models')
    os.environ['DATA_DIR'] = os.path.join(base_dir, 'data')
    os.environ['FEATURES_DIR'] = os.path.join(base_dir, 'data', 'features')
    
    # Log the directories
    logger.info(f"Models directory: {os.environ['MODELS_DIR']}")
    logger.info(f"Data directory: {os.environ['DATA_DIR']}")
    logger.info(f"Features directory: {os.environ['FEATURES_DIR']}")
    
    # Run the server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )

if __name__ == "__main__":
    main()
