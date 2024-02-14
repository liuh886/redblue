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
from func.logger import StructuredLogger, SystemStatus

# Create a FastAPI app
tracker = APIRouter()

# Create a system state with a logger
app = SystemStatus()
app.logger = StructuredLogger('tracker', '/logs')

# Initialize the system state
start_date = "2024-02-13" # os.getenv("START_DATE", "2024-02-10")
app.system_date = datetime.strptime(start_date, '%Y-%m-%d')
app.last_gps_t = app.system_date
app.last_beacon_t = app.system_date
app.updating = True

# Initialize the trains mapping table (Rame table) / train circulations table / Beacn mapping table
app.trains_mapping_table = {}
app.trains_dict = {}
app.beacon_country = ["GB", "FR", "BE", "NL", "DE"]
app.beacon_mapping_table = {}

async def init_tracker():
    print("Application - KF tracker - is starting...")

    # (1) system initialtion. It will fetch and update the train dictionary
    print(f"Background task initialized with date: {app.system_date}")

    # Run beacon_init and train_init concurrently, and then service_init
    await asyncio.gather(
        beacon_init(app),
        train_init(app)
    )
    await service_init(app)

    # (2) Start background task for continuous updates for every 10 seconds.
    # TODO: to trigger an update task by database. 
    # Need configuration from database side, might Askok can help.
    if app.updating:
        #print('Positioning updating...for the following id', trains_dict.keys())
        asyncio.create_task(
            update_train_positions(
                app,
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
        return await get_realtime_train_data(app.trains_dict, rame_id=None)
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
        return await get_realtime_train_data_geolocation(app.trains_dict)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching trains: {e}")


# get train data by train rame id
@tracker.get("/{rame_id}",
             summary="Get real-time data of the train rameid (trainset id).",
             response_model=List[dict])
async def get_train(rame_id: int):
    try:
        return await get_realtime_train_data(app.trains_dict, rame_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Train not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching train {rame_id}: {e}")


# operations for realtime train dict - update, delete
@tracker.post("/{date}",
              summary="Refresh the list of tracking trains by Rame/Circulation table.",
              description="This can catch the updates from circulation table, "
              "without stopping the real-time service. Do this when the train service has updated."
              "The date is in the format 'YYYY-MM-DD'.",
              response_model=dict)
async def update_train(date: str = start_date):
    try:
        await train_init(app)
        await service_init(app)
        system_date = date
        return {"ok": True}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error updating train: {e}") from e


@tracker.delete("/{rame_id}",
                summary="Remove a train from the tracking list.",
                response_model=dict)
async def delete_train(rame_id: int):
    try:
        # remove rame_id from the train list
        await app.trains_dict.pop(rame_id)
        return {"ok": True}
    except KeyError:
        raise HTTPException(status_code=404, detail="Train not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting train {rame_id}: {e}") from e


@tracker.get("/circulations/",
             summary="Get circulations for a given travel date in %Y-%m-%d",
             response_model=List[dict])
async def get_circulations(date: str = Query(default=start_date,
                                             description="Start datetime in the format YYYY-MM-DD")):
    try:
        if date is None:
            # Using today to update the train data
            date = datetime.today().strftime('%Y-%m-%d')
        return await fetch_circulation_info(date)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {ve}") from ve
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching circulations: {e}") from e

@tracker.get("/status/",
             summary="Get the status of the tracker service.",
             response_model=dict)
async def get_status():
    try:
        return {"Train in operaiton (id)": list(app.trains_dict.keys()), 
                "Updating": app.updating,
                "The last timestamp of GPS": app.last_gps_t.isoformat(),
                "The last timestamp of Beacon": app.last_beacon_t.isoformat(),
                "The length of GPS records (last cycle)": app.length_gps,
                "The length of Beacon records (last cycle)": app.length_beacon,
                "The initial date of system": app.system_date.isoformat(),
                "The beacon country code": app.beacon_country,
                "The trains_mapping_table": {obj.rame_id: obj.rame_number for obj in app.trains_mapping_table.values()},
                }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@tracker.get("/logs/",
             summary="Get the logs of the tracker service.",
             response_model=dict)
async def get_logs():
    try:
        # Check if the log file exists
        if not os.path.exists(app.logger.log_dir):
            raise HTTPException(status_code=404, detail="Log file not found.")

        new_logs = []
        with open(app.logger.log_dir, "r") as file:
            for line in file:
                # Assuming each log entry starts with a timestamp in ISO format
                log_time_str = line.split(' - ')[0]
                try:
                    log_time = datetime.fromisoformat(log_time_str)
                    if log_time > app.last_log_query_time:
                        new_logs.append(line.strip())
                except ValueError:
                    # If the log line doesn't start with a timestamp, skip it
                    continue

        # Update the timestamp of the last query
        app.last_log_query_time = datetime.now()

        return {"logs": new_logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e