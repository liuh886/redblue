from databases import Database
from sqlalchemy import create_engine, MetaData
import os

# Environment variables should be loaded at the application's entry point,
# but we use them here to configure the database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/dbname")

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

async def save_processed_data(data):
    query = "INSERT INTO your_table_name(field1, field2, ...) VALUES(:field1, :field2, ...)"
    await database.execute(query=query, values=data.dict())

# You might also include other utility functions here for common database operations
