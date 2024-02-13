# kf_tracker.py

from fastapi import APIRouter, HTTPException, Query
from typing import List
import os
import asyncio
from datetime import datetime
from func.database import fetch_circulation_info
from func.train_service import (
    train_init,
    service_init,
    get_realtime_train_data,
    get_realtime_train_data_geolocation,
    update_train_positions,
    beacon_init
)

# Create a FastAPI app
tracker = APIRouter()

# The train service need to be initialed with a date, to assign service to trains.
# When deployment in future , the start_date should be today().
# Assuming your date is in the format 'YYYY-MM-DD'
start_date = os.getenv("START_DATE", "2024-02-07")
start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')

# The update task in background use the timestamp to query updates from database.
# For initialization, the timestamp is the midnight of the start date.
last_gps_t = start_date_obj
last_beacon_t = start_date_obj

# Initialize the trains
trains_mapping_table = {}
# Initialize the train in service (circulations) dictionary
trains_dict = {}
# The beacon country code
beacon_country = ["GB", "FR", "BE", "NL", "DE"]
beacon_mapping_table = {}
beacon_dict = {}

async def init_tracker():
    print("Application - KF tracker is starting...")

    # (1) system initialtion. It will fetch and update the train dictionary
    print(f"Background task initialized with date: {start_date_obj}")

    # Run beacon_init and train_init concurrently, and then service_init
    await asyncio.gather(
        beacon_init(start_date_obj, beacon_country, beacon_mapping_table),
        train_init(trains_mapping_table)
    )
    await service_init(start_date_obj, trains_dict, trains_mapping_table)

    # (2) Start background task for continuous updates for every 10 seconds.
    # TODO: to trigger an update task by database. Need configuration from database side, might Askok can help.
    asyncio.create_task(
        update_train_positions(
            trains_dict,
            last_gps_t,
            last_beacon_t,
            update_interval=10,
        )
    )

# Endpoints for the KF tracker below


# get all real-time train data
@tracker.get("/",
             summary="Get all real-time train data flow.",
             description="The most recent geolocation after combining ICOMERA and RNE TIS.",
             response_description="Return a list of real-time trains.",
             response_model=List[dict])
async def get_trains():
    try:
        return await get_realtime_train_data(trains_dict, rame_id=None)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching trains: {e}")


# get all geolocation of real-time train data
@tracker.get("/geolocation",
             summary="Get all real-time geolocations, the custom endpoint.",
             response_description="Return a list of real-time trains with less columns.",
             response_model=List[dict],
             deprecated=True)
async def get_trains_geolocation():
    try:
        return await get_realtime_train_data_geolocation(trains_dict)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching trains: {e}")


# get train data by train rame id
@tracker.get("/{rame_id}",
             summary="Get real-time data of the train rameid (trainset id).",
             response_model=List[dict])
async def get_train(rame_id: int):
    try:
        return await get_realtime_train_data(trains_dict, rame_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Train not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching train {rame_id}: {e}")


# operations for realtime train dict - update, delete
@tracker.post("/{date}",
              summary="Refresh the list of tracking trains by Rame/Circulation table.",
              description="This can catch the updates from circulation table, "
              "without stopping the real-time service. Do this when the train service is updated."
              "The date is in the format 'YYYY-MM-DD'. If not provided, today's date is used.",
              response_model=dict)
async def update_train(date: str = start_date):
    try:
        await train_init(trains_mapping_table)
        await service_init(date, trains_dict, trains_mapping_table)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error updating train: {e}") from e


@tracker.delete("/{rame_id}",
                summary="Remove a train from the tracking list.",
                response_model=dict)
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


@tracker.get("/circulations/",
             summary="Get circulations for a given travel date in %Y-%m-%d",
             response_model=List[dict]
             )
async def get_circulations(date: str = Query(default=start_date,
                           description="Start datetime in the format YYYY-MM-DD")):
    try:
        if date is None:
            # Using today to update the train data
            date = datetime.today().strftime('%Y-%m-%d')
        return await fetch_circulation_info(date)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching circulations: {e}")
