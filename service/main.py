from fastapi import FastAPI, HTTPException, Depends
from .database import connect_to_database, disconnect_from_database, save_processed_data
from .models import DataModel, ProcessedDataModel
from .kalman_filter import apply_kalman_filter  # Import your Kalman Filter logic here

app = FastAPI()

@app.on_event("startup")
async def startup():
    await connect_to_database()

@app.on_event("shutdown")
async def shutdown():
    await disconnect_from_database()

@app.post("/process/", response_model=ProcessedDataModel)
async def process_data(data: DataModel):
    try:
        # Apply your Kalman Filter to the incoming data
        filtered_data = apply_kalman_filter(data.value)
        
        # Construct the processed data model
        processed_data = ProcessedDataModel(timestamp=data.timestamp, filtered_value=filtered_data)
        
        # Save the processed data to the database (optional)
        await save_processed_data(processed_data)
        
        return processed_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
