from pydantic import BaseModel
from datetime import datetime


class CirculationInfo(BaseModel):
    RameId: str
    TrainNumber: str
    TravelDate: datetime
    Origin: str
    Destination: str
    Status: str
    # Include other fields from the Circulations table as needed
