# train_service.py

from database import fetch_rame_info, fetch_circulation_info, fetch_gps_update, fetch_train_running_update
from fastapi import HTTPException
import asyncio
from kalman_filter import apply_kalman_filter
from typing import List


class Train:
    def __init__(self,
                 rame_id: int,
                 rame_number: int # trainset number
                 ):
        # identifier
        self.rame_id = rame_id   # id
        self.length = None
        self.rame_number = rame_number  # trainset number
        self.systemid = []
        # service. A trainset can have multiple train service in a day.
        self.train_number = []
        self.traveldate = []
        self.origin = []
        self.destination = []
        self.status = []
        self.type = []
        self.um_train_number = []
        self.um_position = []
        self.total_distance = []
        # positioning signal status. This is the most recent signals.
        self.latitute = None
        self.longitude = None
        self.altitude = None
        self.speed = None
        self.heading = None
        self.lock = None
        self.satellites = None
        self.quality = None
        self.last_gps_t = None
        self.last_gps_code = None  # ICOMERA/Noman system id
        self.last_beacon_t = None
        self.last_beacon_code = None
        # Kalman filter. The most recent Kalman filter results.
        self.kalman_filter = None
        self.kalman_filter_state = None
        self.kalman_filter_gain = None
        self.kalman_filter_measurement = None
        self.kalman_filter_prediction = None
        self.kalman_filter_residual = None

        # Add other attributes as needed.

    def update(self, new_gnss, new_beacons):
        # Method to update the train object with data from GPS or Beacons

        # Handle multiple records. Sort updates by timestamp, and process from old to new

        # TODO: generate update matrix from new_gnss and new_beacons

        # the last states matrix of train
        matrix = 1
        # Kalman filter return a matrix of states
        results = apply_kalman_filter(matrix)

        # Update attributes as needed
        if new_gnss:
            # GNSS metadata
            self.lock = new_gnss[-1]['metaData.lock']
            self.satellites = new_gnss[-1]['metaData.satellites']
            self.quality = new_gnss[-1]['quality']

            # timestamp metadata
            self.last_gps_t = new_gnss[-1]['timestamp']
        
        if new_beacons:
            self.last_beacon_t = new_beacons[-1]['LocationDateTime']
            self.last_beacon_code = new_beacons[-1]['LocationPrimaryCode']

        # Update the last position of train
        self.latitute = results[-1]
        self.longitude = results[-1]
        self.altitude = results[-1]
        self.speed = results[-1]
        self.heading = results[-1]

        return results

    def to_dict(self):
        return {
            "rame_id": self.rame_id,
            "rame_number": self.rame_number,
            "train_number": self.train_number,
            "traveldate": self.traveldate,
            "origin": self.origin,
            "destination": self.destination,
            "status": self.status,
            "type": self.type,
            "um_train_number": self.um_train_number,
            "um_position": self.um_position,
            'last_gps_t': self.last_gps_t,
            'last_beacon_t': self.last_beacon_t,
            'last_beacon_code': self.last_beacon_code,
            'lock': self.lock,
            'satellites': self.satellites,
            'quality': self.quality,
            'latitute': self.latitute,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'speed': self.speed,
            'heading': self.heading,
            'kalman_filter': self.kalman_filter,
            'kalman_filter_state': self.kalman_filter_state,
            'kalman_filter_gain': self.kalman_filter_gain,
            'kalman_filter_prediction': self.kalman_filter_prediction,
            'kalman_filter_residual': self.kalman_filter_residual,
        }

    def to_dict_geo(self):
        return {
            "rame_id": self.rame_id,
            "rame_number": self.rame_number,
            "train_number": self.train_number,
            "traveldate": self.traveldate,
            "origin": self.origin,
            'latitute': self.latitute,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'speed': self.speed,
            'heading': self.heading,
        }


async def train_init(trains_mapping_table):
    
    rame_info = await fetch_rame_info()
    
    if rame_info:
        for item in rame_info:
            # fetch the Rame information for each train.
            train = Train(rame_id=item['Id'],
                          rame_number=item['RameNumber'])
            # TODO: change rules for Noman
            train.systemid.extend([int(str(item['RameNumber'])+'2'), int(str(item['RameNumber'])+'7')])

            # Using id as key, to store in trains_mapping_table
            trains_mapping_table[item['Id']] = train
    else:
        raise HTTPException(
            status_code=404, detail="No data found for the given Rames table")

async def service_init(date, trains_dict, trains_mapping_table):
    # To initialize/update the train service by circulation_data,
    
    # Fetch the Circulation data
    circulation_data = await fetch_circulation_info(date)

    if not circulation_data:
        raise HTTPException(
            status_code=404, detail="No data found in Circulation table for the given date")
    
    # Check service line by line
    for item in circulation_data:

        # if RameId is not Null
        if item['RameId'] != 'NULL' and item['TrainNumber'][-4:].isdigit():
            
            # Change dtypes from the table Circulations
            id = int(item['RameId'])
            train_number = int(item['TrainNumber'][-4:]) # get the last four digits of the train number
            um_train_number = int(item['UMTrainNumber'][-4:]) if item['UMTrainNumber'] != 'NULL' else None

            # If (a) id has been initialized (in mapping table) and (b) The service has not been assign.
            # Then, it is ok to assign the service to the train
            train_obj = trains_dict.get(id)
            train_numbers = getattr(train_obj, 'train_number', [])

            if id in trains_mapping_table.keys() and train_number not in train_numbers:
                
                if train_obj is None:
                    # Add the train to the dictionary of trains in service
                    trains_dict[id] = trains_mapping_table[id]

                # Add the service to the train
                trains_dict[id].train_number.append(train_number)
                trains_dict[id].traveldate.append(item['TravelDate'])
                trains_dict[id].origin.append(item['Origin'])
                trains_dict[id].destination.append(item['Destination'])
                trains_dict[id].status.append(item['Status'])
                trains_dict[id].type.append(item['Type'])
                trains_dict[id].um_train_number.append(um_train_number)
                trains_dict[id].um_position.append(item['UMPosition'])
                trains_dict[id].total_distance.append(item['TotalDistance'])
    print('The following train id is in operation', trains_dict.keys())
    #print([i.to_dict()  for i in trains_dict.values()])

async def update_train_positions(trains_dict, last_gps_t, last_beacon_t, update_interval=10,):

    print(trains_dict)
    while True:
        # Localisation
        gps_updates = await fetch_gps_update(last_gps_t)
        beacon_updates = await fetch_train_running_update(last_beacon_t)

        # Update the train objects with the new data
        for train in trains_dict.values():
            # Filter updates for this train; adjust the filtering logic as needed
            train_icomera_updates = [
                update for update in gps_updates if update['systemid'] in train.systemid]
            train_info_updates = [
                update for update in beacon_updates if update['OTI_OperationalTrainNumber'][-4:] in train.train_number]
            # TODO: waht if OTI_OperationalTrainNumber is not available? 
            # TODO: what if the beacon update comes in delay?
            train.update(train_icomera_updates, train_info_updates)
        if gps_updates:
            last_gps_t = gps_updates[-1]['timestamp']
        if beacon_updates:
            last_beacon_t = beacon_updates[-1]['LocationDateTime']

        # Sleep for some time before fetching updates again
        await asyncio.sleep(update_interval)


async def get_realtime_train_data(trains_dict, rame_id:int =None) -> List:
    # Return the train data as a dictionary
    return [train.to_dict() for train in trains_dict.values() if rame_id is None or train.rame_id == rame_id]


async def get_realtime_train_data_geolocation(trains_dict, rame_id:int =None) -> List:
    # Return the train data as a dictionary
    return [train.to_dict_geo() for train in trains_dict.values() if rame_id is None or train.rame_id == rame_id]
