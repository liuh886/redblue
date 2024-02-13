# replayer.py

from fastapi import APIRouter, HTTPException, Query, Path
from fastapi.responses import StreamingResponse
from typing import List
import asyncio
from datetime import datetime
from func.database import fetch_circulation_info
from func.train_service import (
    train_init,
    service_init,
    historical_train_positions
)

# Create a FastAPI app
replayer = APIRouter()

# Initialize the trains
trains_mapping_table = {}
# Initialize the train in service (circulations) dictionary
trains_dict = {}
initialization_date = None


async def replay_data_generator(
                                pace: int,
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
    start_date: datetime = Query(..., description="Start date for the replay"),
    end_date: datetime = Query(default=None, description="End date for the replay"),
    pace: int = Query(default=1, description="Pace of the replay in seconds"),
    rame_id: int = Query(
        default=None, description="Rame id for the replay")
):

    print(f"Application - Replayer is starting... with date: {start_date}")

    if start_date.date() != initialization_date.date():
        # Reinitialization - Fetch the trian and service information by trains_dict
        await train_init(trains_mapping_table)
        await service_init(start_date, trains_dict, trains_mapping_table)

    if not end_date:
        end_date = start_date + datetime.timedelta(minutes=10)

    # Fetch all historical train data after KF
    result = await historical_train_positions(trains_dict,
                                              start_date,
                                              end_date,
                                              rame_id)
    
    return StreamingResponse(replay_data_generator(
                             pace,
                             result),
                             media_type="text/plain")


@replayer.get("/{rame_id}/",
              summary="Get historical data of the train rameid (trainset id).",
              response_model=List[dict])
async def get_train(
    start_date: datetime = Query(..., description="Start datetime in the format YYYY-MM-DD HH:MM:SS"),
    end_date: datetime = Query(..., description="End datetime in the format YYYY-MM-DD HH:MM:SS"),
    rame_id: int = Path(..., description="The ID of the train")
):
    try:
        if start_date != initialization_date:
        # Reinitialization - Fetch the trian and service information by trains_dict
            await train_init(trains_mapping_table)
            await service_init(start_date, trains_dict, trains_mapping_table)

        return await historical_train_positions(trains_dict,
                                                start_date,
                                                end_date,
                                                rame_id)

    except KeyError:
        raise HTTPException(status_code=404, detail="Train not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching train {rame_id}: {e}")


@replayer.get("/{date}/circulations",
              summary="Get circulations for a given travel date in %Y-%m-%d",
              response_model=List[dict])
async def get_circulations(date: str = Path(..., description="Travel date in format YYYY-MM-DD")):
    try:
        return await fetch_circulation_info(date)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching circulations: {e}") from e