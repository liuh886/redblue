# main.py

from fastapi import FastAPI, HTTPException
from typing import List, Dict
import os
import asyncio
from datetime import datetime
from models import CirculationInfo, TrainModel
from database import fetch_circulation_info, connect_to_database, disconnect_from_database
from train_service import (
    train_init,
    service_init,
    get_realtime_train_data,
    get_realtime_train_data_geolocation,
    update_train_positions
)

# Create a FastAPI app
app = FastAPI()
start_date = os.getenv("START_DATE", "2024-02-06")
# Assuming your date is in the format 'YYYY-MM-DD'

start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
last_gps_t = start_date_obj
last_beacon_t = start_date_obj

# Initialize the trains
trains_mapping_table = {}
# Initialize the train in service (circulations) dictionary
trains_dict = {}

# Start and stop background tasks
@app.on_event("startup")
async def startup_event():
    print("FastAPI application is starting...")
    # Establish database connection
    await connect_to_database()

    # (1) system initialtion. It will fetch and update the train dictionary
    print(f"Background task initialized with date: {start_date_obj}")
    asyncio.create_task(train_init(trains_mapping_table))
    asyncio.create_task(service_init(start_date_obj, trains_dict, trains_mapping_table))

    # (2) Start background task for continuous updates for each train
    asyncio.create_task(
        update_train_positions(
            trains_dict,
            last_gps_t,
            last_beacon_t,
            update_interval=10,
        )
    )


@app.on_event("shutdown")
async def shutdown_event():
    await disconnect_from_database()
    print("FastAPI application is shutting down...")

# get all real-time train data


@app.get("/realtime/",
         tags=["realtime"],
         summary="Get all real-time train data flow.",
         description="The most recent geolocation after combining ICOMERA and RNE TIS.",
         response_description="The dictionary of all real-time train data.",
         response_model=List[dict])
async def get_trains():
    try:
        return await get_realtime_train_data(trains_dict, rame_id=None)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching trains: {e}")

# get all geolocation of real-time train data


@app.get("/realtime/geolocation",
         tags=["realtime_geolocation"],
         summary="Get all real-time geolocations, the custom endpoint.",
         response_model=List[dict])
async def get_trains_geolocation():
    try:
        return await get_realtime_train_data_geolocation(trains_dict)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching trains: {e}")

# get train data by train rame id


@app.get("/realtime/{rame_id}",
         tags=["realtime_trainset"],
         summary="Get real-time train data of train rameid (trainset id).",
         response_model=List[dict])
async def get_train(rame_id: int):
    try:
        return await get_realtime_train_data(trains_dict, rame_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Train not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching train {rame_id}: {e}")

# operations for train dict - update, delete


@app.post("/realtime/{date}",
          tags=["realtime_reinitialation"],
          summary="Refresh the list of tracking trains by circulation table.",
          response_model=List)
async def update_train(date: str = None):
    try:
        if date is None:
            # Using today to update the train data
            date = datetime.datetime.now().strftime("%Y-%m-%d")
        elif isinstance(date, datetime):
            date = date.strftime("%Y-%m-%d")
        await service_init(date, trains_dict, trains_mapping_table)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error updating train: {e}") from e


@app.delete("/realtime/{rame_id}", response_model=List)
async def delete_train(rame_id: int):
    global trains_dict
    try:
        # remove rame_id from the train list
        await trains_dict.pop(rame_id)
        return {"ok": True}
    except KeyError:
        raise HTTPException(status_code=404, detail="Train not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting train {rame_id}: {e}") from e

# return the circulations data


@app.get("/realtime/circulations", response_model=List[dict])
async def get_circulations():
    try:
        # Using today to update the train data
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        return await fetch_circulation_info(date)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching circulations: {e}") from e

# Future endpoints for replay and query functionalities can be added here