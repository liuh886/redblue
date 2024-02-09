from databases import Database
from sqlalchemy import create_engine, MetaData
import os

# Environment variables should be loaded at the application's entry point,
# but we use them here to configure the database connection

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/dbname")
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
    await database.connect()

async def disconnect_from_database():
    await database.disconnect()


# Update by timestamp
## ICOMERA Table Operations
async def fetch_icomera_update(last_timestamp=None):
    # Construct the query to fetch only records newer than the last_timestamp
    query = """
    SELECT systemid, timestamp, latitude, longitude, altitude, speed, heading, metaData.lock, metaData.satellites, quality
    FROM {DB_SCHEMA}.ICOMERA
    WHERE timestamp > :last_timestamp
    ORDER BY timestamp ASC
    """
    return await database.fetch_all(query=query, values={"last_timestamp": last_timestamp})

## TrainRunningInformation_MainTable_4005 Operations
async def fetch_train_running_update(last_timestamp=None):
    # Construct the query similarly
    query = """
    SELECT OTI_OperationalTrainNumber, ROTN_OTI_OperationalTrainNumber, LocationPrimaryCode, LocationDateTime, ReferencedLocationDateTime, MessageDataTime
    FROM {DB_SCHEMA}.TrainRunningInformation_MainTable_4005
    WHERE LocationDateTime > :last_timestamp
    ORDER BY LocationDateTime ASC
    """
    return await database.fetch_all(query=query, values={"last_timestamp": last_timestamp})

# Update by train number
async def fetch_icomera_data(systemid):
    query = """
    SELECT systemid, timestamp, latitude, longitude, altitude, speed, heading, metaData.lock, metaData.satellites, quality
    FROM {DB_SCHEMA}.ICOMERA
    WHERE systemid = :systemid
    """
    return await database.fetch_all(query=query, values={"systemid": systemid})

async def fetch_train_running_info(oti_operational_train_number):
    query = """
    SELECT OTI_OperationalTrainNumber, ROTN_OTI_OperationalTrainNumber, LocationPrimaryCode, LocationDateTime, ReferencedLocationDateTime, MessageDataTime
    FROM {DB_SCHEMA}.TrainRunningInformation_MainTable_4005
    WHERE OTI_OperationalTrainNumber = :oti_operational_train_number OR ROTN_OTI_OperationalTrainNumber = :oti_operational_train_number
    """
    return await database.fetch_all(query=query, values={"oti_operational_train_number": oti_operational_train_number})

# Initial fetch

# Rame Table Operations. Fetch by Id
async def fetch_rame_info(rame_id):
    query = """
    SELECT Id, Ramenumber
    FROM {DB_SCHEMA}.Rame
    WHERE Id = :rame_id
    """
    return await database.fetch_one(query=query, values={"rame_id": rame_id})

# Circulations Table fetch by TravelDate
async def fetch_circulation_info(travel_date):
    query = """
    SELECT RameId, Status, TrainNumber, TravelDate, Origin, Destination, Type, UMTrainNumber, UMPosition
    FROM {DB_SCHEMA}.Circulations
    WHERE TravelDate = :travel_date
    """
    return await database.fetch_all(query=query, values={"travel_date": travel_date})

# PrimaryLocation Operations. Fetch by LocationCode
async def fetch_primary_location(location_code):
    query = """
    SELECT Latitude, Longitude 
    FROM {DB_SCHEMA}.PrimaryLocation
    WHERE LocationCode = :location_code
    """
    return await database.fetch_one(query=query, values={"location_code": location_code})

# TODO: Think about what need to be return to database
async def save_processed_data(data):
    query = "INSERT INTO your_table_name(field1, field2, ...) VALUES(:field1, :field2, ...)"
    await database.execute(query=query, values=data.dict())