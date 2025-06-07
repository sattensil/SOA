"""
FastAPI application for mine safety injury rate prediction.
"""
import os
import sys
import logging
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator, metrics

# Create a global instrumentator instance
instrumentator = Instrumentator()

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

# Instrument the app with Prometheus
# This should be done after FastAPI app initialization and before adding routes/middleware.
# The global 'instrumentator' instance (defined at the top) is used here.
instrumentator.instrument(app)  # Adds default metrics like request counts, latency by path, etc.

# Add custom-named standard metrics
instrumentator.add(metrics.request_size(metric_name="mine_safety_api_request_size_bytes"))
instrumentator.add(metrics.response_size(metric_name="mine_safety_api_response_size_bytes"))
instrumentator.add(metrics.latency(metric_name="mine_safety_api_latency_seconds")) # Already a default, but this ensures our naming if needed
instrumentator.add(metrics.requests(metric_name="mine_safety_api_requests_total")) # Already a default, but this ensures our naming if needed

# Import our custom metrics registry
from .metrics import custom_registry, initialize_metrics

# Create a custom metrics endpoint that combines both registries
@app.get("/metrics", tags=["monitoring"])
async def metrics():
    # Generate metrics from both the default registry and our custom registry
    default_metrics = generate_latest()
    custom_metrics = generate_latest(custom_registry)
    
    # Combine the metrics
    combined_metrics = default_metrics + custom_metrics
    
    return Response(content=combined_metrics, media_type=CONTENT_TYPE_LATEST)

logger.info("Prometheus metrics instrumentation initialized and /metrics endpoint exposed.")

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
    # Log startup information
    logger.info("Starting Mine Safety Injury Rate Prediction API")
    logger.info(f"Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    
    # Initialize metrics with default labels
    initialize_metrics()
    logger.info("Metrics initialized with default labels")
    
    # Log model directories
    models_dir = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))
    data_dir = os.environ.get('DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
    features_dir = os.environ.get('FEATURES_DIR', os.path.join(data_dir, 'features'))
    
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Features directory: {features_dir}")
    
    # Handle MLflow if available
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

# Import metrics from metrics.py
from .metrics import model_version_counter, prediction_latency

@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information."""
    logger.info("Shutting down Mine Safety Injury Rate Prediction API")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
