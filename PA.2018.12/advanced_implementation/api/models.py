"""
Pydantic models for the FastAPI application.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any

class MineData(BaseModel):
    """Input data for a single mine."""
    MINE_ID: str = Field(..., description="Mine ID")
    MINE_NAME: Optional[str] = Field(None, description="Mine name")
    CURRENT_STATUS: str = Field(..., description="Current status of the mine (e.g., 'ACTIVE')")
    CURRENT_MINE_TYPE: Optional[str] = Field(None, description="Current mine type")
    CURRENT_MINE_STATUS: Optional[str] = Field(None, description="Current mine status")
    CURRENT_CONTROLLER_ID: Optional[str] = Field(None, description="Current controller ID")
    CURRENT_OPERATOR_ID: Optional[str] = Field(None, description="Current operator ID")
    STATE: str = Field(..., description="State where the mine is located")
    FIPS_CNTY_CD: Optional[str] = Field(None, description="FIPS county code")
    LONGITUDE: Optional[float] = Field(None, description="Longitude coordinate")
    LATITUDE: Optional[float] = Field(None, description="Latitude coordinate")
    PRIMARY: Optional[str] = Field(None, description="Primary commodity")
    SECONDARY: Optional[str] = Field(None, description="Secondary commodity")
    CURRENT_MINE_SUBUNIT: Optional[str] = Field(None, description="Current mine subunit")
    SUBUNIT_CD: Optional[str] = Field(None, description="Subunit code")
    CAL_YR: int = Field(..., description="Calendar year")
    YEAR: int = Field(..., description="Year (required for validation/model)")
    AVG_EMPLOYEE_CNT: float = Field(..., description="Average employee count")
    HOURS_WORKED: float = Field(..., description="Hours worked")
    COAL_PRODUCTION: Optional[float] = Field(None, description="Coal production")
    AVG_EMPLOYEE_CNT_UNDERGROUND: Optional[float] = Field(None, description="Average employee count underground")
    UNDERGROUND_HOURS: Optional[float] = Field(None, description="Underground hours")
    AVG_EMPLOYEE_CNT_SURFACE: Optional[float] = Field(None, description="Average employee count surface")
    SURFACE_HOURS: Optional[float] = Field(None, description="Surface hours")
    INJURIES_COUNT: Optional[int] = Field(0, description="Number of injuries")
    
    class Config:
        schema_extra = {
            "example": {
                "MINE_ID": "1234567",
                "MINE_NAME": "Mine Name",
                "CURRENT_STATUS": "ACTIVE",
                "CURRENT_MINE_TYPE": "Surface",
                "CURRENT_MINE_STATUS": "Active",
                "CURRENT_CONTROLLER_ID": "Controller ID",
                "CURRENT_OPERATOR_ID": "Operator ID",
                "STATE": "WV",
                "FIPS_CNTY_CD": "01001",
                "LONGITUDE": 40.0,
                "LATITUDE": -80.0,
                "PRIMARY": "Coal",
                "SECONDARY": "Metal",
                "CURRENT_MINE_SUBUNIT": "Subunit",
                "SUBUNIT_CD": "Subunit Code",
                "CAL_YR": 2022,
                "YEAR": 2022,
                "AVG_EMPLOYEE_CNT": 45.0,
                "HOURS_WORKED": 90000.0,
                "COAL_PRODUCTION": 100000.0,
                "AVG_EMPLOYEE_CNT_UNDERGROUND": 20.0,
                "UNDERGROUND_HOURS": 40000.0,
                "AVG_EMPLOYEE_CNT_SURFACE": 25.0,
                "SURFACE_HOURS": 50000.0,
                "INJURIES_COUNT": 5
            }
        }

class BatchMineData(BaseModel):
    """Batch of mine data for prediction."""
    mines: List[MineData]
    
    class Config:
        schema_extra = {
            "example": {
                "mines": [
                    {
                        "MINE_ID": "1234567",
                        "MINE_NAME": "Mine Name",
                        "CURRENT_STATUS": "ACTIVE",
                        "CURRENT_MINE_TYPE": "Surface",
                        "CURRENT_MINE_STATUS": "Active",
                        "CURRENT_CONTROLLER_ID": "Controller ID",
                        "CURRENT_OPERATOR_ID": "Operator ID",
                        "STATE": "WV",
                        "FIPS_CNTY_CD": "01001",
                        "LONGITUDE": 40.0,
                        "LATITUDE": -80.0,
                        "PRIMARY": "Coal",
                        "SECONDARY": "Metal",
                        "CURRENT_MINE_SUBUNIT": "Subunit",
                        "SUBUNIT_CD": "Subunit Code",
                        "CAL_YR": 2022,
                        "AVG_EMPLOYEE_CNT": 45.0,
                        "HOURS_WORKED": 90000.0,
                        "COAL_PRODUCTION": 100000.0,
                        "AVG_EMPLOYEE_CNT_UNDERGROUND": 20.0,
                        "UNDERGROUND_HOURS": 40000.0,
                        "AVG_EMPLOYEE_CNT_SURFACE": 25.0,
                        "SURFACE_HOURS": 50000.0,
                        "INJURIES_COUNT": 5
                    },
                    {
                        "MINE_ID": "7654321",
                        "MINE_NAME": "Mine Name 2",
                        "CURRENT_STATUS": "ACTIVE",
                        "CURRENT_MINE_TYPE": "Underground",
                        "CURRENT_MINE_STATUS": "Active",
                        "CURRENT_CONTROLLER_ID": "Controller ID 2",
                        "CURRENT_OPERATOR_ID": "Operator ID 2",
                        "STATE": "CO",
                        "FIPS_CNTY_CD": "01003",
                        "LONGITUDE": 40.0,
                        "LATITUDE": -80.0,
                        "PRIMARY": "Metal",
                        "SECONDARY": "Coal",
                        "CURRENT_MINE_SUBUNIT": "Subunit 2",
                        "SUBUNIT_CD": "Subunit Code 2",
                        "CAL_YR": 2022,
                        "AVG_EMPLOYEE_CNT": 120.0,
                        "HOURS_WORKED": 240000.0,
                        "COAL_PRODUCTION": 200000.0,
                        "AVG_EMPLOYEE_CNT_UNDERGROUND": 40.0,
                        "UNDERGROUND_HOURS": 80000.0,
                        "AVG_EMPLOYEE_CNT_SURFACE": 30.0,
                        "SURFACE_HOURS": 60000.0,
                        "INJURIES_COUNT": 10
                    }
                ]
            }
        }

class PredictionResponse(BaseModel):
    """Response for a single prediction."""
    mine_id: str
    predicted_injury_rate: float
    
    class Config:
        schema_extra = {
            "example": {
                "mine_id": "1234567",
                "predicted_injury_rate": 0.0865
            }
        }
    
class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    predictions: List[PredictionResponse]
    model_version: str
    average_predicted_rate: float
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "mine_id": "1234567",
                        "predicted_injury_rate": 0.0865
                    },
                    {
                        "mine_id": "7654321",
                        "predicted_injury_rate": 0.1234
                    }
                ],
                "model_version": "enhanced_xgboost_v1.0",
                "average_predicted_rate": 0.10495
            }
        }

class ModelInfo(BaseModel):
    """Model information."""
    model_type: str
    model_version: str
    feature_count: int
    top_features: List[str]
    objective: str
    last_updated: str
    
    class Config:
        schema_extra = {
            "example": {
                "model_type": "XGBoost",
                "model_version": "enhanced_xgboost_v1.0",
                "feature_count": 151,
                "top_features": ["HOURS_WORKED", "AVG_EMPLOYEE_CNT", "YEAR", "PRIMARY_Coal", "CURRENT_MINE_TYPE_Surface"],
                "objective": "reg:squarederror",
                "last_updated": "2025-06-01"
            }
        }

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    feature_count: int
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "preprocessor_loaded": True,
                "feature_count": 15
            }
        }

# Model Version Management Models
class ModelVersionMetrics(BaseModel):
    """Metrics for a model version."""
    test_rmse: Optional[float] = None
    test_mae: Optional[float] = None
    test_ll: Optional[float] = None


class ModelVersionInfo(BaseModel):
    """Information about a model version."""
    version: str
    created_at: str
    status: str
    is_active: bool
    metrics: ModelVersionMetrics
    description: Optional[str] = None


class ModelVersionsResponse(BaseModel):
    """Response containing all model versions."""
    versions: List[ModelVersionInfo]
    active_version: str


class ModelVersionActivateResponse(BaseModel):
    """Response for activating a model version."""
    message: str


class ActiveModelVersionResponse(BaseModel):
    """Response containing the active model version."""
    active_version: str
