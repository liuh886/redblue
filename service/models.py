from pydantic import BaseModel
from datetime import datetime

class DataModel(BaseModel):
    timestamp: datetime
    value: float
    # Add other relevant fields for your data

class ProcessedDataModel(BaseModel):
    timestamp: datetime
    filtered_value: float  # Assuming the Kalman Filter processes a 'value'
    # Add additional fields as needed
