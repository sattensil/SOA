"""
FastAPI application for mine safety injury rate prediction.
"""
import os
import sys
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_client import Counter, Histogram

# Add the parent directory to the path so we can import the scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import routes
from .routes import router

# Import MLflow utilities if available
try:
    from scripts.mlflow_utils import start_mlflow_server, get_active_model_version
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(
    title="Mine Safety Injury Rate Prediction API",
    description="""API for predicting injury rates in mines based on historical data.
    
    ## Model Versioning
    
    This API supports MLflow-based model versioning. You can:
    - List all available model versions via `/model-versions`
    - Get the active model version via `/model-versions/active`
    - Set a specific model version as active via `/model-versions/{version}/activate`
    - Use a specific model version for prediction by passing the `version` query parameter to prediction endpoints
    
    ## Prediction Endpoints
    
    - `/predict` - Predict injury rate for a single mine
    - `/predict-batch` - Predict injury rates for multiple mines
    
    ## Monitoring
    
    The API includes Prometheus metrics at `/metrics` for monitoring request counts, latency, and errors.
    """,
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
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

# Try to start MLflow server on startup if available
@app.on_event("startup")
async def startup_event():
    if MLFLOW_AVAILABLE:
        try:
            # Get the active model version to log it
            active_version = get_active_model_version()
            logger.info(f"Starting API with active model version: {active_version}")
            
            # Try to start MLflow server in the background
            start_mlflow_server()
            logger.info("MLflow server started in the background")
        except Exception as e:
            logger.warning(f"Could not start MLflow server: {str(e)}")
            logger.info("API will continue without MLflow server")
    else:
        logger.info("MLflow integration not available")

    logger.info("Starting Mine Safety Injury Rate Prediction API")
    logger.info(f"Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    
    # Log model directories
    models_dir = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))
    data_dir = os.environ.get('DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
    features_dir = os.environ.get('FEATURES_DIR', os.path.join(data_dir, 'features'))
    
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Features directory: {features_dir}")

# Set up Prometheus metrics
instrumentator = Instrumentator()

# Add default metrics
instrumentator.instrument(app)

# Add custom metrics
# Request size
instrumentator.add(
    metrics.request_size(metric_name="mine_safety_api_request_size_bytes")
)

# Response size
instrumentator.add(
    metrics.response_size(metric_name="mine_safety_api_response_size_bytes")
)

# Latency by endpoint and method
instrumentator.add(
    metrics.latency(metric_name="mine_safety_api_latency_seconds")
)

# Add exception metrics
instrumentator.add(
    metrics.requests(metric_name="mine_safety_api_requests_total")
)

# Add model version metrics counter
model_version_counter = Counter(
    "mine_safety_api_model_version_total",
    "Number of predictions by model version",
    ["version"]
)

# Add prediction latency histogram by model version
prediction_latency = Histogram(
    "mine_safety_api_prediction_latency_seconds",
    "Prediction latency by model version",
    ["version"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

# Initialize metrics
instrumentator.expose(app)
instrumentator.expose(app, endpoint="/metrics", include_in_schema=True, tags=["monitoring"])

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
