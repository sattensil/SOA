"""
FastAPI application for mine safety injury rate prediction.
"""
import os
import sys
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add the parent directory to the path so we can import the scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Import routes
from .routes import router

# Initialize FastAPI app
app = FastAPI(
    title="Mine Safety Injury Rate Prediction API",
    description="API for predicting injury rates in mines based on MSHA data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Include routes
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("Starting Mine Safety Injury Rate Prediction API")
    logger.info(f"Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    
    # Log model directories
    models_dir = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))
    data_dir = os.environ.get('DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
    features_dir = os.environ.get('FEATURES_DIR', os.path.join(data_dir, 'features'))
    
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Features directory: {features_dir}")

@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information."""
    logger.info("Shutting down Mine Safety Injury Rate Prediction API")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
