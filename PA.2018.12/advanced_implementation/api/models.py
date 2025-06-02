"""
Pydantic models for the FastAPI application.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union

class MineData(BaseModel):
    """Input model for a single mine's data."""
    MINE_ID: str = Field(..., description="Mine ID")
    YEAR: int = Field(..., description="Year of the data")
    PRIMARY: str = Field(..., description="Primary commodity")
    CURRENT_MINE_TYPE: str = Field(..., description="Current mine type")
    CURRENT_STATUS: str = Field(..., description="Current status of the mine")
    CURRENT_MINE_SUBTYPE: Optional[str] = Field(None, description="Current mine subtype")
    FIPS_CNTY: str = Field(..., description="FIPS county code")
    AVG_EMPLOYEE_CNT: float = Field(..., description="Average employee count")
    HOURS_WORKED: float = Field(..., description="Hours worked")
    COAL_METAL_IND: str = Field(..., description="Coal or metal indicator")
    INJURIES_COUNT: Optional[float] = Field(None, description="Count of injuries (optional for prediction)")
    INJURY_RATE: Optional[float] = Field(None, description="Injury rate (optional for prediction)")
    
    class Config:
        schema_extra = {
            "example": {
                "MINE_ID": "1234567",
                "YEAR": 2016,
                "PRIMARY": "Coal",
                "CURRENT_MINE_TYPE": "Surface",
                "CURRENT_STATUS": "Active",
                "CURRENT_MINE_SUBTYPE": "Strip",
                "FIPS_CNTY": "01001",
                "AVG_EMPLOYEE_CNT": 45.0,
                "HOURS_WORKED": 90000.0,
                "COAL_METAL_IND": "C",
                "INJURIES_COUNT": None,
                "INJURY_RATE": None
            }
        }

class BatchMineData(BaseModel):
    """Input model for batch prediction."""
    mines: List[MineData]
    
    class Config:
        schema_extra = {
            "example": {
                "mines": [
                    {
                        "MINE_ID": "1234567",
                        "YEAR": 2016,
                        "PRIMARY": "Coal",
                        "CURRENT_MINE_TYPE": "Surface",
                        "CURRENT_STATUS": "Active",
                        "CURRENT_MINE_SUBTYPE": "Strip",
                        "FIPS_CNTY": "01001",
                        "AVG_EMPLOYEE_CNT": 45.0,
                        "HOURS_WORKED": 90000.0,
                        "COAL_METAL_IND": "C",
                        "INJURIES_COUNT": None,
                        "INJURY_RATE": None
                    },
                    {
                        "MINE_ID": "7654321",
                        "YEAR": 2016,
                        "PRIMARY": "Metal",
                        "CURRENT_MINE_TYPE": "Underground",
                        "CURRENT_STATUS": "Active",
                        "CURRENT_MINE_SUBTYPE": "Drift",
                        "FIPS_CNTY": "01003",
                        "AVG_EMPLOYEE_CNT": 120.0,
                        "HOURS_WORKED": 240000.0,
                        "COAL_METAL_IND": "M",
                        "INJURIES_COUNT": None,
                        "INJURY_RATE": None
                    }
                ]
            }
        }

class PredictionResponse(BaseModel):
    """Response model for a single prediction."""
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
    """Response model for batch predictions."""
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
    """Model for model information."""
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
    """Model for health check response."""
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
                "feature_count": 151
            }
        }
