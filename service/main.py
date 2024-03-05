# main.py
from fastapi import FastAPI
from apps.tracker import tracker, init_tracker
from apps.replayer import replayer, test_replayer
from func.database import connect_to_database, disconnect_from_database
import asyncio
import logging

profiler = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# connect with database
@app.on_event("startup")
async def startup_event():
    await connect_to_database()
    # initialize data
    asyncio.create_task(init_tracker())
    #asyncio.create_task(test_replayer())

@app.on_event("shutdown")
async def shutdown_event():
    await disconnect_from_database()
    print("-----FastAPI application is shutting down...")

# Include the routers
app.include_router(tracker, prefix="/realtime", tags=["Real-time tracker"])
app.include_router(replayer, prefix="/replay", tags=["Replayer"])

@app.get("/", response_model=dict)
async def root():
    return {"message": "Welcome to RedBlue API"}
