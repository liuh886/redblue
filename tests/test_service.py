# test_service.py

import sys
from pathlib import Path
# Add the root of the project to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pytest
from httpx import AsyncClient
from datetime import datetime, timedelta
from service.main import app 
from service.train_service import Train
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_get_trains():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/trains/")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)  # Assuming the endpoint returns a dictionary of trains

@pytest.mark.asyncio
async def test_get_circulations():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/circulations/")
    print(response.json())
    assert response.status_code == 200
    assert isinstance(response.json(), dict)  # Assuming the endpoint returns a dictionary of trains


@pytest.mark.asyncio
async def test_get_train_by_id():
    train_id = 22  # Use an actual train ID that exists in your test dataset

    mock_train = Train(
        rame_id=train_id,
        train_number="123",
        rame_number="456",
        traveldate=datetime.now(),
        origin="London",
        destination="Manchester",
        status="Delayed",
        type="Passenger",
        um_train_number="789",
        um_position="1A"
    )

    # Mock the database call to return mock_train when queried by train_id
    with patch("database.fetch_rame_info", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = mock_train

        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get(f"/trains/{train_id}")
        
        assert response.status_code == 200
        assert response.json()['rame_id'] == str(train_id)  # Ensure the response matches the mock train data

@pytest.mark.asyncio
async def test_delete_train():
    train_id = 22  # Use an actual train ID that exists in your test dataset
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.delete(f"/trains/{train_id}")
    assert response.status_code == 200
    # You might want to add additional checks to ensure the train was indeed deleted
