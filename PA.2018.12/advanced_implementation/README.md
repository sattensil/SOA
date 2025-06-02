# Advanced XGBoost Implementation for Mine Safety Injury Rate Prediction

This project implements an advanced XGBoost model for predicting injury rates in mines based on the MSHA Mine Data from 2013-2016. The implementation includes a FastAPI service for serving predictions via a REST API, containerized with Docker for easy deployment.

## Project Structure

```
advanced_implementation/
├── api/                # FastAPI service implementation
│   ├── main.py         # FastAPI application setup
│   ├── models.py       # Pydantic models for request/response validation
│   ├── routes.py       # API endpoints implementation
│   └── utils.py        # Utility functions for the API
├── data/               # Data directory for processed data
│   ├── features/       # Preprocessed features and artifacts
│   └── processed_data.csv # Cleaned data ready for modeling
├── models/             # Directory for saved models
│   ├── enhanced_feature_names.joblib  # Feature names for the model
│   ├── enhanced_xgboost_model.joblib  # Serialized XGBoost model
│   └── enhanced_xgboost_model.json    # XGBoost model in JSON format
├── results/            # Directory for results and visualizations
├── run_api.py          # Script to run the FastAPI server
├── scripts/            # Python scripts for the pipeline
│   ├── data_loader.py                 # Data loading and basic cleaning
│   ├── feature_engineering.py         # Feature engineering and preprocessing
│   ├── model_training.py              # XGBoost model training and evaluation
│   ├── main.py                        # Main script to run the standard pipeline
│   ├── enhanced_feature_engineering.py # Enhanced feature engineering with debugging
│   ├── enhanced_model_training.py     # Enhanced model training with debugging
│   ├── enhanced_main.py               # Main script for the enhanced pipeline
│   └── prediction_utility.py          # Reusable prediction utility module
├── simple_prediction.py   # Simplified prediction utility for the API
├── Dockerfile            # Docker configuration for containerization
├── docker-compose.yml    # Docker Compose configuration
└── tests/              # Test directory
    ├── integration/    # Integration tests
    │   ├── test_api.py # API integration tests
    │   ├── test_end_to_end_pipeline.py # End-to-end pipeline tests
    │   └── test_model_prediction.py # Model prediction tests
    └── unit/           # Unit tests
        ├── test_data_loader.py # Data loader tests
        ├── test_enhanced_feature_engineering.py # Feature engineering tests
        ├── test_enhanced_main.py # Main script tests
        └── test_enhanced_model_training.py # Model training tests
```

## Getting Started

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- Poetry (for dependency management)

### Installation

1. Clone the repository
2. Install dependencies with Poetry:
   ```
   poetry install
   ```

### Running the API

#### Local Development

To run the FastAPI service locally for development:

```bash
python run_api.py
```

This will start the FastAPI server on port 8080. You can access the API documentation at http://localhost:8080/docs.

#### Docker Deployment

To run the API in a Docker container:

```bash
docker-compose up -d
```

This will build and start the Docker container with the API running on port 8080.

### Command Line Arguments

- `--skip-data-loading`: Skip data loading and use existing processed data
- `--skip-feature-engineering`: Skip feature engineering and use existing features
- `--n-estimators`: Number of estimators for XGBoost (default: 100)
- `--max-depth`: Maximum depth of trees for XGBoost (default: 5)
- `--learning-rate`: Learning rate for XGBoost (default: 0.1)
- `--random-state`: Random state for reproducibility (default: 42)

## Pipeline Steps

### 1. Data Loading

The `data_loader.py` script handles:
- Loading raw data from CSV
- Basic data cleaning
- Creating the target variable (injury rate per 2000 hours)
- Filtering out closed mines and mines with low employee hours
- Validating record counts against the exam solution (27 rows removed due to missing values)

### 2. Feature Engineering

The `feature_engineering.py` and `enhanced_feature_engineering.py` scripts handle:
- Feature transformation (e.g., log transformation)
- Feature creation (e.g., combining mine type and commodity)
- Categorical encoding
- Numerical scaling
- Train-test splitting with stratification on binned injury rates
- Special handling for multi-label PRIMARY field using dummy variables
- Detailed logging of record counts and feature creation

### 3. Model Training

The `model_training.py` and `enhanced_model_training.py` scripts handle:
- Training an XGBoost model with Poisson objective
- Evaluating model performance using log-likelihood, RMSE, and MAE
- Saving the trained model
- Plotting feature importance with proper handling of feature name mismatches

## Enhanced Implementation Features

The enhanced implementation includes several improvements over the standard pipeline:

1. **Extensive Logging**: Detailed logging at each step of the pipeline to track record counts and data transformations
2. **Validation Checks**: Verification that data cleaning steps match the expected results from the exam solution
3. **Improved Stratification**: Proper handling of NaN values in injury rate bins for stratified sampling
4. **PRIMARY Field Processing**: Special handling for the multi-label PRIMARY field with detailed logging
5. **Feature Name Tracking**: Accurate tracking of feature names through the preprocessing pipeline
6. **Error Handling**: Robust handling of potential mismatches in feature names and importances

## Performance Metrics

The enhanced model achieves the following metrics:

- Train RMSE: 0.0843
- Test RMSE: 0.0865
- Train MAE: 0.0350
- Test MAE: 0.0354

## Testing Framework

The project includes a comprehensive testing framework with both unit tests and integration tests:

### Unit Tests

Unit tests focus on testing individual components of the pipeline in isolation:

- **Data Loading Tests**: Verify that data cleaning correctly removes closed mines and mines with low employee hours
- **Feature Engineering Tests**: Ensure feature engineering creates proper train/test dictionaries with features, targets, and weights
- **Model Training Tests**: Validate model training and evaluation metrics calculation
- **Main Script Tests**: Verify command-line argument parsing and pipeline orchestration

Run unit tests with:

```bash
python -m unittest discover -s advanced_implementation/tests/unit -v
```

### Integration Tests

Integration tests validate the end-to-end functionality of the pipeline:

- **End-to-End Pipeline Test**: Tests the complete pipeline from raw data to saved model and metrics
- **Prediction Tests**: Verify the ability to load a saved model and make predictions on new data
- **Model Consistency Tests**: Ensure predictions are consistent for the same input
- **Batch Prediction Tests**: Validate that the model can handle multiple samples at once

The integration tests use temporary directories and synthetic datasets to isolate from production. They include robust fallback mechanisms to create missing artifacts during testing.

#### API Integration Tests

API integration tests validate the FastAPI endpoints:

- **Root Endpoint Test**: Verifies the root endpoint returns correct API information
- **Health Check Test**: Ensures the health check endpoint returns proper status
- **Model Info Test**: Validates the model metadata endpoint
- **Prediction Endpoint Test**: Tests single prediction functionality with input validation
- **Batch Prediction Test**: Verifies batch prediction capabilities

The API tests use FastAPI's TestClient to test endpoints without running a server.

Run integration tests with:

```bash
python -m pytest advanced_implementation/tests/integration -v
```

## API Implementation

The project includes a FastAPI implementation for serving model predictions, containerized with Docker for easy deployment:

### API Endpoints

- **GET /** - Root endpoint with basic API information
- **GET /health** - Health check endpoint
- **GET /model/info** - Get model metadata and information
- **GET /model/features** - Get list of features used by the model
- **POST /predict** - Predict injury rate for a single mine
- **POST /predict/batch** - Predict injury rates for multiple mines

### API Features

1. **Input Validation**: Robust validation using Pydantic models
2. **Error Handling**: Comprehensive error handling with appropriate HTTP status codes
3. **Documentation**: Interactive API documentation with Swagger UI
4. **Batch Processing**: Support for batch predictions
5. **Model Info**: Endpoints for model metadata and feature information
6. **Field Mapping**: Automatic mapping of field names (e.g., CURRENT_STATUS to MINE_STATUS)
7. **Missing Value Handling**: Graceful handling of missing fields in prediction requests

### Field Mappings

The API supports the following field mappings to make it more flexible:

- `CURRENT_STATUS` → `MINE_STATUS`: The API will automatically map the mine status field
- `INJURIES_COUNT` → `NUM_INJURIES`: For injury count data
- `HOURS_WORKED` → `EMP_HRS_TOTAL`: For employee hours worked data

This allows the API to accept data in different formats while maintaining compatibility with the underlying model.

### Docker Deployment

The API is containerized using Docker for easy deployment:

```bash
# Build the Docker image
docker build -t mine-safety-api .

# Run the container
docker run -d -p 8081:8080 --name mine-safety-container \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/data/features:/app/features" \
  -e MODELS_DIR=/app/models \
  -e DATA_DIR=/app/data \
  -e FEATURES_DIR=/app/features \
  mine-safety-api
```

Access the interactive API documentation at http://localhost:8081/docs

## Future Enhancements

Potential enhancements for the API:

1. **Monitoring**: Add performance tracking and feature distribution monitoring
2. **Model Versioning**: Implement MLflow for experiment tracking
3. **Reliability Features**: Add circuit breakers and fallback mechanisms
4. **Load Testing**: Ensure the system can handle production traffic
5. **CI/CD Pipeline**: Automate testing and deployment
