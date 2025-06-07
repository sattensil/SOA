# Advanced XGBoost Implementation for Mine Safety Injury Rate Prediction

This project implements an advanced XGBoost model for predicting injury rates in mines based on the MSHA Mine Data from 2013-2016. The implementation includes a FastAPI service for serving predictions via a REST API, containerized with Docker for easy deployment.

## Data Versioning with DVC

This project uses [DVC](https://dvc.org/) for data version control. DVC tracks the `advanced_implementation/data` directory and stores data versions in a local remote directory (`../dvc_local_storage`). This keeps your Git repo lightweight and enables reproducible experiments.

### DVC Setup
- DVC initialized in project as a subdirectory repo
- `advanced_implementation/data` tracked by DVC (not Git)
- Local DVC remote: `../dvc_local_storage`

### Typical Workflow

#### 1. Restore Data (after fresh clone or checkout)
```bash
poetry run dvc pull
```
This restores the latest version of `advanced_implementation/data` from the local DVC remote.

#### 2. Update Data and Track Changes
- Add or modify files in `advanced_implementation/data`
- Track changes:
```bash
poetry run dvc add advanced_implementation/data
```
- Commit DVC metadata to Git:
```bash
git add advanced_implementation/data.dvc
```
- Push data to remote:
```bash
poetry run dvc push
```

#### 3. Ignore Data in Git
The `.gitignore` is automatically updated to exclude data files. Only DVC metadata (`data.dvc`) is tracked in Git.

#### 4. Check Data Status
```bash
poetry run dvc status
```

### Notes
- The local DVC remote (`../dvc_local_storage`) is suitable for testing and small teams. For collaboration or cloud backup, configure a remote like S3, GDrive, or Azure.
- To remove or roll back data, use DVC commands to ensure reproducibility.


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

### Data Processing Improvements
1. **Extensive Logging**: Detailed logging at each step of the pipeline to track record counts and data transformations
2. **Validation Checks**: Verification that data cleaning steps match the expected results from the exam solution
3. **Improved Stratification**: Proper handling of NaN values in injury rate bins for stratified sampling
4. **PRIMARY Field Processing**: Special handling for the multi-label PRIMARY field with detailed logging
5. **Feature Name Tracking**: Accurate tracking of feature names through the preprocessing pipeline
6. **Error Handling**: Robust handling of potential mismatches in feature names and importances

### Advanced Feature Engineering
1. **Interaction Terms**: 
   - Employee count × underground hours (`EMP_X_UNDERGROUND`)
   - Employee count × strip mining hours (`EMP_X_STRIP`)
   - Employee count × surface mining hours (`EMP_X_SURFACE`)
   - Employee count × mill hours (`EMP_X_MILL`)
2. **Polynomial Features**: Square of employee count (`AVG_EMP_TOTAL_SQ`) to capture non-linear relationships
3. **Ratio Features**: Hours per employee (`HRS_PER_EMP`) to measure employee utilization intensity
4. **Risk Score**: Combined weighted risk factors from different mining activities, scaled by employee count
5. **Binary Flags**: 
   - High underground mining operations (`HIGH_UNDERGROUND`)
   - Large operations (`LARGE_OPERATION`)
6. **Combined Categorical Features**: Mine type and commodity combinations (`MINE_CHAR`)

### Modeling Enhancements
1. **Two-Stage Approach**: 
   - Classification stage to predict if injuries will occur
   - Regression stage to predict the number of injuries (only for mines predicted to have injuries)
2. **ADASYN Oversampling**:
   - Adaptive Synthetic Sampling (ADASYN) applied to the injury regressor training data
   - Balances injury classes by generating synthetic samples for minority classes
   - Dynamically adjusts n_neighbors parameter based on smallest class size
   - Maps synthetic samples back to realistic injury counts
   - Significantly improves accuracy for mines with 1 injury
3. **Hyperparameter Optimization**: 
   - Bayesian optimization for hyperparameter tuning
   - Early stopping to prevent overfitting
4. **Sample Weighting**: 
   - Higher weights for mines with more injuries to improve rare class prediction
5. **Threshold Optimization**: 
   - Optimized classification threshold to match the official solution's accuracy for the majority class
   - This ensures we maintain high accuracy on "no injury" cases while improving minority class detection
6. **Ensemble Methods**: 
   - XGBoost's built-in ensemble capabilities
   - Gradient boosting with regularization to prevent overfitting

## Performance Metrics

The enhanced model (version 2.1.0) achieves the following regression metrics:

- Train RMSE: 0.0843
- Test RMSE: 0.0865
- Train MAE: 0.0350
- Test MAE: 0.0354

## Comparison with Official Solution

### Data Distribution

Both models work with a highly imbalanced dataset:

- Mines with 0 injuries: 28,654 (79.3%)
- Mines with 1 injury: 3,950 (10.9%)
- Mines with 2 injuries: 1,344 (3.7%)
- Mines with 3+ injuries: 2,172 (6.0%)

### Model Performance Comparison

#### Official Solution (Poisson GLM)

The official R model achieves:
- Class 0 (No injuries) accuracy: 90.4%
- Class 1 (1 injury) accuracy: 28.5%
- Class 2 (2 injuries) accuracy: 11.3%
- Class 3+ (3+ injuries) accuracy: 60.7%
- Overall accuracy: 78.6%

#### Advanced Implementation (XGBoost with ADASYN, v3.0.0)

Our advanced implementation with ADASYN (version 3.0.0) achieves:
- Class 0 (No injuries) accuracy: 90.7% (maintained at similar level)
- Class 1 (1 injury) accuracy: 59.0% (107% improvement over R model)
- Class 2 (2 injuries): Limited samples in test set
- Class 3+ (3+ injuries): Limited samples in test set
- Overall binary accuracy: 84.1% (5.5% improvement over R model)

### Key Differences

| Aspect | Official Solution | Advanced Implementation |
|--------|------------------|-------------------------|
| Model Type | Poisson GLM | Two-stage XGBoost (Classification + Regression) |
| Feature Engineering | Basic interactions and log transformations | Enhanced with interaction terms, polynomial features, ratio features, risk scores, and binary flags |
| Class Weighting | None | Injury-weighted with exponent of 2.5 |
| Threshold Optimization | Fixed threshold | Fine-grained threshold search with injury-weighted metrics |
| Zero-Inflation Handling | Single model | Two-stage approach for better handling of zeros |

### Business Impact

The improvements in our advanced implementation with ADASYN translate to significant business value:

1. **Better Injury Prediction**: Correctly identifying over 100% more mines with 1 injury compared to the R model
2. **Enhanced Risk Management**: More accurate identification of high-risk operations with a 3.3% improvement in identifying mines with 3+ injuries
3. **Targeted Interventions**: Better allocation of safety resources to mines most likely to experience injuries
4. **Regulatory Compliance**: Improved ability to meet safety requirements and avoid penalties
5. **Overall Performance**: 0.9% improvement in overall accuracy across all mines

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

## Model Versioning with MLflow

This project uses MLflow for model versioning and tracking. The implementation enables:

1. **Model Registry**: Tracking and versioning of different model iterations
2. **Model Serving**: Serving specific model versions via the API
3. **Model Metrics**: Tracking prediction counts by model version using Prometheus

### Creating New Model Versions

To create and register new model versions:

1. Use the enhanced training pipeline in `scripts/enhanced_main.py`:
   ```bash
   poetry run python scripts/enhanced_main.py --register-model
   ```

2. This will:
   - Load and process the data
   - Engineer features
   - Train a new model
   - Register the model with MLflow
   - Assign a new version number

3. Set a specific model version as active:
   ```bash
   poetry run python scripts/enhanced_main.py --set-active-version <version_number>
   ```

### Key Files for Model Versioning

If you clone this repository and want to work with model versions, these are the essential files:

- `scripts/enhanced_main.py` - Main entry point for training and registering models
- `scripts/mlflow_utils.py` - Utilities for MLflow integration
- `scripts/data_loader.py` - Data loading and preprocessing
- `scripts/enhanced_feature_engineering.py` - Feature engineering pipeline
- `scripts/enhanced_model_training.py` - Model training with MLflow tracking

### Monitoring Model Version Usage

The API tracks prediction counts by model version using Prometheus metrics:

1. The metric `mine_safety_api_model_version_total` counts predictions by version
2. View these metrics in Prometheus at http://localhost:9090
3. Visualize in Grafana dashboards at http://localhost:3000

### Testing Model Versions

To test model version metrics and generate dashboard data:

```bash
# Generate metrics data with different model versions
poetry run python generate_metrics_data.py

# Test model version metrics specifically
poetry run python test_model_version_metrics.py
```

## Future Enhancements

Potential enhancements for the API:

1. **Monitoring**: Enhanced performance tracking and feature distribution monitoring
2. **Reliability Features**: Add circuit breakers and fallback mechanisms
3. **Load Testing**: Ensure the system can handle production traffic
4. **CI/CD Pipeline**: Automate testing and deployment
5. **A/B Testing**: Compare performance of different model versions
