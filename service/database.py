# database.py

from databases import Database
from sqlalchemy import create_engine, MetaData
import os
from typing import Optional, List
from datetime import datetime

# Environment variables should be loaded at the application's entry point,
# but we use them here to configure the database connection

DB_CONNECTION = os.getenv("DB_CONNECTION")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_DATABASE = os.getenv("DB_DATABASE")
DATABASE_URL = f"{DB_CONNECTION}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"
DB_SCHEMA = os.getenv("DB_SCHEMA")

# For asynchronous operation
database = Database(DATABASE_URL)

# For synchronous operation, like with Alembic migrations
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Utility functions to connect and disconnect from the database
async def connect_to_database():
    print(f"Connecting to database")
    await database.connect()

async def disconnect_from_database():
    print(f"Disconnecting from database")
    await database.disconnect()

# Update by timestamp
## ICOMERA Table Operations TODO: Support Noman 
async def fetch_gps_update(last_timestamp: datetime)-> List[dict]:
    # Construct the query to fetch only records newer than the last_timestamp
    query = f"""
    SELECT "systemid", "timestamp", "latitude", "longitude", "altitude", "speed", "heading", "quality", "metaData.satellites", "metaData.lock"
    FROM "{DB_SCHEMA}"."icomera"
    WHERE "timestamp" > :last_timestamp
    ORDER BY "timestamp" ASC
    """
    return await database.fetch_all(query=query, values={"last_timestamp": last_timestamp})

## TrainRunningInformation_MainTable_4005 Operations TODO: Support UK Rail network
async def fetch_train_running_update(last_timestamp: datetime)-> List[dict]:
    # Construct the query similarly
    query = f"""
    SELECT "OTI_OperationalTrainNumber", "ROTN_OTI_OperationalTrainNumber", "LocationPrimaryCode", "LocationDateTime", "ReferencedLocationDateTime", "MessageDateTime"
    FROM "{DB_SCHEMA}"."TrainRunningInformation_MainTable_4005"
    WHERE "LocationDateTime" > :last_timestamp
    ORDER BY "LocationDateTime" ASC
    """
    return await database.fetch_all(query=query, values={"last_timestamp": last_timestamp})

# Update by train id / number
async def fetch_gps(systemid: int)-> List[dict]:
    query = f"""
    SELECT "systemid", "timestamp", "latitude", "longitude", "altitude", "speed", "heading", "quality", "metaData.satellites", "metaData.lock"
    FROM "{DB_SCHEMA}"."icomera"
    WHERE "systemid" = :systemid
    ORDER BY "timestamp" ASC
    """
    return await database.fetch_all(query=query, values={"systemid": systemid})

async def fetch_train_running(oti_operational_train_number: str)-> List[dict]:
    query = f"""
    SELECT "OTI_OperationalTrainNumber", "ROTN_OTI_OperationalTrainNumber", "LocationPrimaryCode", "LocationDateTime", "ReferencedLocationDateTime", "MessageDataTime"
    FROM "{DB_SCHEMA}"."TrainRunningInformation_MainTable_4005"
    WHERE "OTI_OperationalTrainNumber" = :oti_operational_train_number OR "ROTN_OTI_OperationalTrainNumber" = :oti_operational_train_number
    ORDER BY "LocationDateTime" ASC
    """
    return await database.fetch_all(query=query, values={"oti_operational_train_number": oti_operational_train_number})

# Initial fetch
# Rame Table Operations.
async def fetch_rame_info() -> List[dict]:
    query = f"""
    SELECT "Id", "RameNumber"
    FROM "{DB_SCHEMA}"."Rames"
    """
    return await database.fetch_all(query=query)

# Circulations Table fetch by TravelDate
async def fetch_circulation_info(date: datetime.date) -> List[dict]:
    '''
    Fetch all circulations for a given travel date.
    
    TODO: The problem rightnow: 
    (1)There are many dupilcated records in the database. 
    A trainset can have multiple train service in a day.
    (2) Null value in the RameId columns. 
    '''

    query = f"""
    SELECT "RameId", "Status", "TrainNumber", "TravelDate", "Origin", "Destination", "Type", "UMTrainNumber", "UMPosition", "TotalDistance"
    FROM "{DB_SCHEMA}"."Circulations"
    WHERE "TravelDate"::date = :travel_date
    """
    return await database.fetch_all(query=query, values={"travel_date": date})

# PrimaryLocation Operations. Fetch by LocationCode
async def fetch_primary_location(location_code: int) -> Optional[dict]:
    query = f"""
    SELECT "Latitude", "Longitude"
    FROM "{DB_SCHEMA}"."PrimaryLocation"
    WHERE "LocationCode" = :location_code
    """
    return await database.fetch_all(query=query, values={"location_code": location_code})

# TODO: Think about what need to be return to database
async def save_processed_data(data):
    query = "INSERT INTO your_table_name(field1, field2, ...) VALUES(:field1, :field2, ...)"
    await database.execute(query=query, values=data.dict())