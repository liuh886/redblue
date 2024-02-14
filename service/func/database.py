# database.py

from databases import Database
from sqlalchemy import create_engine, MetaData
import os
from typing import Optional, List
from datetime import datetime
from typing import Union

# Environment variables are used here to configure the database connection
# Make sure .env is in the root directory of the project, and ignore it in .gitignore

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
async def fetch_beacon_update(last_timestamp: datetime)-> List[dict]:
    # Construct the query similarly
    query = f"""
    SELECT "OTI_OperationalTrainNumber", "ROTN_OTI_OperationalTrainNumber", "LocationPrimaryCode", "LocationDateTime", "ReferencedLocationDateTime", "MessageDateTime"
    FROM "{DB_SCHEMA}"."TrainRunningInformation_MainTable_4005"
    WHERE "LocationDateTime" > :last_timestamp
    ORDER BY "LocationDateTime" ASC
    """
    return await database.fetch_all(query=query, values={"last_timestamp": last_timestamp})


# Query historical by train id / date
async def fetch_gps(systemid_list: list, start: datetime, end: datetime)-> List[dict]:
    # Convert the list of systemids into a string format suitable for SQL IN clause
    systemid_str = ",".join(f"'{item}'" for item in systemid_list)

    query = f"""
    SELECT "systemid", "timestamp", "latitude", "longitude", "altitude", "speed", "heading", "quality", "metaData.satellites", "metaData.lock"
    FROM "{DB_SCHEMA}"."icomera"
    WHERE "systemid" IN ({systemid_str}) AND "timestamp" BETWEEN :start AND :end
    ORDER BY "timestamp" ASC
    """
    return await database.fetch_all(query=query, values={"start": start, "end": end})

async def fetch_beacon(train_number_list: list, start: datetime, end: datetime)-> List[dict]:
    # Convert the list of train numbers into a string format suitable for SQL IN clause
    train_number_str = ",".join(f"'{item}'" for item in train_number_list)
    
    query = f"""
    SELECT "OTI_OperationalTrainNumber", "ROTN_OTI_OperationalTrainNumber", "LocationPrimaryCode", "LocationDateTime", "ReferencedLocationDateTime", "MessageDateTime"
    FROM "{DB_SCHEMA}"."TrainRunningInformation_MainTable_4005"
    WHERE ("OTI_OperationalTrainNumber" IN ({train_number_str}) OR "ROTN_OTI_OperationalTrainNumber" IN ({train_number_str}))
    AND "LocationDateTime" BETWEEN :start AND :end
    ORDER BY "LocationDateTime" ASC
    """
    return await database.fetch_all(query=query, values={"start": start, "end": end})

# Initial fetch
# Rame Table Operations.
async def fetch_rame_info() -> List[dict]:
    query = f"""
    SELECT "Id", "RameNumber"
    FROM "{DB_SCHEMA}"."Rames"
    """
    return await database.fetch_all(query=query)

# Circulations Table fetch by TravelDate
async def fetch_circulation_info(date: Union[datetime, str]) -> List[dict]:
    '''
    Fetch all circulations for a given travel date.
    
    TODO: The problem rightnow: 
    (1)There are many dupilcated records in the database. 
    A trainset can have multiple train service in a day.
    (2) Null value in the RameId columns. 
    '''

    # Format the date as a string in 'YYYY-MM-DD' format
    if not isinstance(date, datetime):
        date = datetime.strptime(date, '%Y-%m-%d')

    query = f"""
    SELECT "RameId", "Status", "TrainNumber", "TravelDate", "Origin", "Destination", "Type", "Positioning", "UMTrainNumber", "UMPosition", "TotalDistance"
    FROM "{DB_SCHEMA}"."Circulations"
    WHERE "TravelDate"::date = :travel_date
    """
    return await database.fetch_all(query=query, values={"travel_date": date})

# PrimaryLocation Operations.
async def fetch_primary_location(date: datetime, beacon_country: list) -> List[dict]:
    '''
    Commets: The database table is not well designed.

    End_Validity is a text.

    '''
    country_code_str = ",".join(f"'{item}'" for item in beacon_country)

    query = f"""
    SELECT "Latitude", "Longitude", "Primary_Location_Code", "Start_Validity", "Country_ISO_code"
    FROM "{DB_SCHEMA}"."PrimaryLocation"
    WHERE (CAST("End_Validity" AS DATE) > :date OR "End_Validity" IS NULL)
    AND "Country_ISO_code" IN ({country_code_str})
    """
    return await database.fetch_all(query=query, values={"date": date})

# TODO: Think about what need to be return to database
async def save_processed_data(data):
    query = "INSERT INTO your_table_name(field1, field2, ...) VALUES(:field1, :field2, ...)"
    await database.execute(query=query, values=data.dict())