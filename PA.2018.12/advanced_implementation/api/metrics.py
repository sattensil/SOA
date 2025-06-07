"""
Prometheus metrics for the Mine Safety Injury Rate Prediction API.
"""
from prometheus_client import Counter, Histogram, CollectorRegistry, REGISTRY
import logging

# Create a custom registry to avoid conflicts with the FastAPI instrumentator
custom_registry = CollectorRegistry()

# Create Prometheus metrics objects with the custom registry
model_version_counter = Counter(
    "mine_safety_api_model_version_total",
    "Number of predictions by model version",
    ["version"],
    registry=custom_registry
)

# Add prediction latency histogram by model version
prediction_latency = Histogram(
    "mine_safety_api_prediction_latency_seconds",
    "Prediction latency by model version",
    ["version"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
    registry=custom_registry
)

# We'll let the FastAPI instrumentator handle these metrics
# to avoid duplication
# request_counter and request_latency are now managed by the instrumentator

# Initialize metrics with default labels
def initialize_metrics():
    """Pre-initialize metrics with default labels to ensure they appear in Prometheus."""
    try:
        # Pre-initialize model version counter with common versions
        versions = ["1", "2", "3", "4", "5"]
        for version in versions:
            # Initialize with 0 to make sure the metric appears in Prometheus
            model_version_counter.labels(version=version)
        
        logging.info("Metrics initialized with default labels")
    except Exception as e:
        logging.error(f"Error initializing metrics: {e}")

# Initialize metrics when this module is imported
initialize_metrics()
