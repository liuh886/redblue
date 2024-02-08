import pytest
from httpx import AsyncClient
from service.main import app  # Replace 'your_fastapi_app' with the actual import path of your FastAPI app

@pytest.mark.asyncio
async def test_process_data():
    # Define mock data input according to your DataModel schema
    mock_data = {
        "timestamp": "2023-01-01T00:00:00",
        "value": 123.45,
        # Include other fields as necessary
    }

    # Use httpx.AsyncClient to make asynchronous requests to your FastAPI app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/process/", json=mock_data)

    # Validate the response
    assert response.status_code == 200
    response_data = response.json()
    assert "filtered_value" in response_data  # Replace 'filtered_value' with the actual field name in your response model
    # Add more assertions as necessary to validate the response content

    # Optionally, verify database interactions, such as ensuring data was saved correctly
    # This might involve querying your test database and comparing the results with expected values
