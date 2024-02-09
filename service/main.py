from datetime import date
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from .database import connect_to_database, disconnect_from_database
from .database import fetch_icomera_update, fetch_train_running_update, fetch_circulation_info_by_date, fetch_icomera_data, fetch_train_running_info
# This is a Pydantic model you should define in models.py
from .models import CirculationInfo
from .train import Train
# Make sure you have this function in database.py
from .database import fetch_rame_info
import asyncio
from fastapi import BackgroundTasks

# Global variable to store train instances
trains_dict = {}
# Example of global variables to track last fetched timestamps
last_icomera_update = None
last_train_info_update = None
# Global variable to keep track of the background task status
is_update_task_running = False

# time of today midnigt
# The start timestamp of the system
today = date.today()
system_timestamp = datetime(today.year, today.month, today.day, 0, 0, 0)

app = FastAPI()


@app.on_event("startup")
async def startup():
    await connect_to_database()


@app.on_event("shutdown")
async def shutdown():
    await disconnect_from_database()



async def fetch_and_update_trains():
    global is_update_task_running,last_icomera_update,last_train_info_update
    while is_update_task_running:
        # logic to fetch updates and apply the Kalman Filter
        # Step 1: Fetch new data from ICOMERA and TIS
        icomera_updates = await fetch_icomera_update(last_icomera_update)
        train_info_updates = await fetch_train_running_update(last_train_info_update)

        # Inside start_process or another appropriate function
        for train in trains_dict.values():
            # Filter updates for this train; adjust the filtering logic as needed
            train_icomera_updates = [update for update in icomera_updates if update['systemid'] == train.rame_id]
            train_info_updates = [update for update in train_info_updates if update['OTI_OperationalTrainNumber'] == train.train_number]
            train.update_with_new_data(train_icomera_updates, train_info_updates)
        
        # Update the last known timestamps
        if icomera_updates:
            last_icomera_update = icomera_updates[-1]['timestamp']
        if train_info_updates:
            last_train_info_update = train_info_updates[-1]['LocationDateTime']

        await asyncio.sleep(30)  # Update rate. Adjust the sleep time as needed


@app.get("/start-process/{process_date}", response_model=List[CirculationInfo])
async def start_process(process_date: system_timestamp):
    try:
        circulation_data = await fetch_circulation_info_by_date(process_date)
        if not circulation_data:
            raise HTTPException(
                status_code=404, detail="No data found for the given date")

        # Train object initialtion
        trains_dict = []
        for item in circulation_data:
            rame_info = await fetch_rame_info(item['RameId'])
            train = Train(rame_id=item['RameId'],
                          train_number=item['TrainNumber'],
                          rame_number=rame_info['Ramenumber'],
                          traveldate=item['TravelDate'],
                          origin=item['Origin'],
                          destination=item['Destination'],
                          status=item['Status'],
                          type=item['Type'],
                          um_train_number=item['UMTrainNumber'],
                          um_position=item['UMPosition'])

            trains_dict.append(train)

        # At this point, you have a list of Train objects initialized with data from Circulations and Rame tables
        # You might store these train objects in a global list, a database, or another suitable storage for later use

        # Location initialtion
        icomera_updates = await fetch_icomera_update(process_date)
        train_info_updates = await fetch_train_running_update(process_date)

        for train in trains_dict:
            train.update_with_new_data(icomera_updates, train_info_updates)

        return [train.to_dict() for train in trains_dict.values()]   # return a summary of initialized Train objects
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start-updates/")
async def start_updates(background_tasks: BackgroundTasks):
    global is_update_task_running
    if not is_update_task_running:
        is_update_task_running = True
        background_tasks.add_task(fetch_and_update_trains)
        return {"message": "Update process started."}
    else:
        return {"message": "Update process is already running."}

@app.post("/stop-updates/")
async def stop_updates():
    global is_update_task_running
    if is_update_task_running:
        is_update_task_running = False
        return {"message": "Update process will be stopped shortly."}
    else:
        return {"message": "Update process is not running."}
