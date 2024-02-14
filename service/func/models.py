from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List


class CirculationInfo(BaseModel):
    RameId: str
    TrainNumber: str
    TravelDate: datetime
    Origin: str
    Destination: str
    Status: str
    # Include other fields from the Circulations table as needed


class TrainModel(BaseModel):
    rame_id: int
    rame_number: int
    train_number: List[int]
    traveldate: List[datetime]
    origin: List[str]
    destination: List[str]
    status: List[str]
    type: List[str]
    um_train_number: List[str] = None
    um_position: List[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = None
    lock: Optional[bool] = None
    satellites: Optional[int] = None
    quality: Optional[str] = None
    last_gnss_update: Optional[datetime] = None
    last_beacon_update: Optional[datetime] = None
    last_beacon_code: Optional[str] = None

    class Config:
        orm_mode = True
