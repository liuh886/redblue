# replayer.py

from fastapi import APIRouter, HTTPException, Query, Path
from fastapi.responses import StreamingResponse
import os
from typing import List
import asyncio
from datetime import datetime, timedelta
from func.database import fetch_circulation_info
from func.train_service import (
    train_init,
    beacon_init,
    service_init,
    historical_train_positions
)
from func.system import SystemLogger,SystemStatus
import pytz
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
import json

# Create a FastAPI app
replayer = APIRouter()

# Create a system state with a logger, a kalman filter
sys = SystemStatus(query_step = timedelta(minutes=30),
                   updating = True,
                   beacon_country = ["GB", "FR", "BE", "NL", "DE"],
                   local_timezone = pytz.timezone('Europe/Paris'),
                   system_timezone = pytz.timezone('UTC'))

sys.logger = SystemLogger('replayer', '/logs')


async def test_replayer():
    print('Test replayer')

    # mock
    start_date = datetime.strptime('2024-02-06 10:00:00', '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime('2024-02-06 10:15:00', '%Y-%m-%d %H:%M:%S')
    rame_id = 13

    sys.system_date = start_date
    # Reinitialization - Fetch the trian and service information by trains_dict
    await asyncio.gather(
        beacon_init(sys),
        train_init(sys)
    )
    await service_init(sys)
    print('-----Starting quering for:', rame_id)
    return await historical_train_positions(rame_id,
                                            sys)

async def replay_data_generator(pace: int,
                                result: dict):
    # Example: just yielding a message every few seconds.

    # Each piece of data that is yielded by the server-side generator function
    # will be sent to the client as soon as it's available.
    for index, row in result.iterrows():
        await asyncio.sleep(pace)
        yield row.to_json() + "\n"  # Convert each row to JSON string and yield


@replayer.get("/feed/",
              summary="Simulate real-time train data flow.",
              description="This endpoint simulates the real-time train data flow by replaying historical data."
              "For each query, the KF process a piece of raw data happened between the start and end date."
              "The pace parameter is used to control the speed of the replay. The default pace is 1 second per record.",
              response_description="The stream feed of slected train.")
async def start_replay(
    start_date: datetime = Query(default='2024-02-06 10:00:00', description="Start date (UTC) for the replay"),
    end_date: datetime = Query(default='2024-02-06 10:15:00', description="End date (UTC) for the replay"),
    pace: int = Query(default=2, description="Pace of the replay in seconds"),
    rame_id: int = Query(
        default=13, description="Rame id for the replay")
):

    sys.system_date = pytz.utc.localize(start_date)
    sys.end_timestamp = pytz.utc.localize(end_date)

    print(f"-----Application - Replayer - is starting... with date: {start_date}")


    # Reinitialization - Fetch the trian and service information by trains_dict
    await asyncio.gather(
    beacon_init(sys),
    train_init(sys)
    )
    await service_init(sys)

    if not end_date:
        end_date = sys.system_date + sys.query_step

    # Fetch all historical train data after KF
    result = await historical_train_positions(rame_id,
                                              sys)

    return StreamingResponse(replay_data_generator(
                             pace,
                             result),
                             media_type="text/plain")

@replayer.get("/{rame_id}/",
              summary="Get historical data of the train rameid (trainset id) in json.")
async def get_train(
    start_date: datetime = Query(default='2024-02-06 10:00:00', description="Start datetime (UTC) in the format YYYY-MM-DD HH:MM:SS"),
    end_date: datetime = Query(default='2024-02-06 10:20:00', description="End datetime (UTC) in the format YYYY-MM-DD HH:MM:SS"),
    rame_id: int = Path(default=13, description="The ID of the train")
):

    try:
        print(f"-----Application - Replayer - is starting... with date: {start_date}")

        sys.system_date = pytz.utc.localize(start_date)
        sys.end_timestamp = pytz.utc.localize(end_date)        
        
        # Reinitialization - Fetch the trian and service information by trains_dict
        await asyncio.gather(
            beacon_init(sys),
            train_init(sys)
        )
        await service_init(sys)

        print('-----Starting quering for (rame_id):', rame_id)

        return await historical_train_positions(rame_id, sys)


    except KeyError:
        raise HTTPException(status_code=404, detail="Train not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching (train id) {rame_id}: {e}")

@replayer.get("/{date}/circulations",
              summary="Get circulations for a given travel date in %Y-%m-%d",
              response_model=List[dict])
async def get_circulations(date: str = Path(..., description="Travel date in format YYYY-MM-DD")):
    try:
        return await fetch_circulation_info(date)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching circulations: {e}") from e


@replayer.get("/logs/",
             summary="Get the logs of the replayer service.",
             response_model=dict)
async def get_logs():
    try:
        # Check if the log file exists
        if not os.path.exists(sys.logger.log_dir):
            raise HTTPException(status_code=404, detail="Log file not found")

        new_logs = []
        with open(sys.logger.log_dir, "r") as file:
            for line in file:
                # Assuming each log entry starts with a timestamp in ISO format
                log_time_str = line.split(' - ')[0]
                try:
                    log_time = datetime.fromisoformat(log_time_str)
                    if log_time > sys.last_log_query_time:
                        new_logs.append(line.strip())
                except ValueError:
                    # If the log line doesn't start with a timestamp, skip it
                    continue

        # Update the timestamp of the last query
        sys.last_log_query_time = datetime.now()

        return {"logs": new_logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@replayer.get("/download-data/", summary="Download the Repalyer's processed data as a file.")
async def download_data():
    # Assuming 'processed_data.csv' is your large dataset file
    file_path = "path/to/your/processed_data.csv"
    return FileResponse(path=file_path, filename="processed_data.csv", media_type='text/csv')
