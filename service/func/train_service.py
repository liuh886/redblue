# train_service.py

from func.database import fetch_rame_info, fetch_primary_location, fetch_circulation_info, fetch_gps_update, fetch_beacon_update, fetch_gps, fetch_beacon
from fastapi import HTTPException
import asyncio
from func.kalman_filter import apply_kalman_filter
from typing import List
from datetime import datetime


class Train:
    def __init__(self,
                 rame_id: int,
                 rame_number: int  # trainset number
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
        self.invert = []
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

    def reset_services(self):
        '''
        Reset the service-related attributes of the train object.
        '''
        self.train_number = []
        self.traveldate = []
        self.origin = []
        self.destination = []
        self.status = []
        self.type = []
        self.invert = []
        self.um_train_number = []
        self.um_position = []
        self.total_distance = []

    def update(self, new_gnss, new_beacons):
        '''
        Method to update the train object with data from GPS or Beacons
        '''

        # Handle multiple records. Sort updates by timestamp, and process from old to new

        # TODO: generate update matrix from new_gnss and new_beacons
        print('length of new_gnss:', len(new_gnss))
        # the last states matrix of train
        
        matrix_old = [[self.latitute, self.longitude, self.altitude, self.speed, self.heading]]

        # Kalman filter return a matrix of states
        results = apply_kalman_filter(matrix_old)
        # Update attributes as needed
        if new_gnss:
            # GNSS metadata
            self.last_gps_code = new_gnss[-1]['systemid']
            self.lock = new_gnss[-1]['metaData.lock']
            self.satellites = new_gnss[-1]['metaData.satellites']
            self.quality = new_gnss[-1]['quality']

            # timestamp metadata
            self.last_gps_t = new_gnss[-1]['timestamp']

        if new_beacons:
            self.last_beacon_t = new_beacons[-1]['LocationDateTime']
            self.last_beacon_code = new_beacons[-1]['LocationPrimaryCode']

        # Update the last position of train
        # .item() is to convert the numpy types to native Python types that are JSON serializable.
        self.latitute = results['filtered'].iloc[-1].item()
        self.longitude = results['filtered'].iloc[-1].item()
        self.altitude = results['filtered'].iloc[-1].item()
        self.speed = results['filtered'].iloc[-1].item()
        self.heading = results['filtered'].iloc[-1].item()

        return results

    def to_dict(self):
        return {
            "rame_id": self.rame_id,
            "rame_number": self.rame_number,
            'systemid': self.systemid,
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


async def train_init(trains_mapping_table: dict) -> None:
    '''
    Initialize the trains_mapping_table with the Rame information from the database.
    '''

    rame_info = await fetch_rame_info()

    if rame_info:
        for item in rame_info:

            # fetch the Rame information for each train.
            train = Train(rame_id=item['Id'],
                          rame_number=item['RameNumber'])
            # TODO: change rules for Noman
            train.systemid.extend(
                [int(str(item['RameNumber'])+'2'), int(str(item['RameNumber'])+'7')])

            # If the train is already in the mapping table, then it will be updated, esle it will be added.
            if item['Id'] in trains_mapping_table:
                trains_mapping_table[item['Id']].rame_id = train.rame_id
                trains_mapping_table[item['Id']
                                     ].rame_number = train.rame_number
                trains_mapping_table[item['Id']].systemid = train.systemid
            else:
                trains_mapping_table[item['Id']] = train
    else:
        raise HTTPException(
            status_code=404, detail="No data found for the given Rames table")

async def beacon_init(start_date_obj: datetime, beacon_country: list, beacon_mapping_table: dict) -> None:
    '''
    Initialize the beacon table with the beacon information from the database.
    '''

    info = await fetch_primary_location(start_date_obj, beacon_country)

    if not info:
        raise HTTPException(
            status_code=404, detail="No data found for the given beacon table")

    for i in info:

        beacon_mapping_table['primary_code'] = i['PrimaryLocationCode']
        beacon_mapping_table['country'] = i['Country_ISO_code']
        beacon_mapping_table['latitude'] = i['Latitude']
        beacon_mapping_table['longitude'] = i['Longitude']



async def service_init(date: datetime, trains_dict: dict, trains_mapping_table: dict) -> None:
    # Fetch the Circulation data for the given date
    circulation_data = await fetch_circulation_info(date)

    if not circulation_data:
        raise HTTPException(
            status_code=404, detail="No data found in Circulation table for the given date")

    # Reset train services based on new circulation data
    for id in trains_dict.keys():
        train_obj = trains_dict[id]
        # Reset service-related attributes for the train. The position-related attributes are not reset.
        train_obj.reset_services()

    # Process each item in the circulation data
    for item in circulation_data:
        # Only process valid records where RameId is not NULL
        if item['RameId'] != 'NULL' and item['TrainNumber'][-4:].isdigit():
            id = int(item['RameId'])

            # Ensure the train object exists in trains_dict
            if id not in trains_dict and id in trains_mapping_table:
                trains_dict[id] = trains_mapping_table[id]

            if id in trains_dict:
                train_obj = trains_dict[id]
                train_number = int(item['TrainNumber'][-4:])
                um_train_number = int(
                    item['UMTrainNumber'][-4:]) if item['UMTrainNumber'] != 'NULL' else None

                # Append new service to the train if it's not already listed
                if train_number not in train_obj.train_number:
                    train_obj.train_number.append(train_number)
                    train_obj.traveldate.append(item['TravelDate'])
                    train_obj.origin.append(item['Origin'])
                    train_obj.destination.append(item['Destination'])
                    train_obj.status.append(item['Status'])
                    train_obj.type.append(item['Type'])
                    train_obj.um_train_number.append(
                        um_train_number if um_train_number is not None else '')
                    train_obj.um_position.append(item['UMPosition'])
                    train_obj.total_distance.append(item['TotalDistance'])

    print('Monitoring the following train ids in operation:', trains_dict.keys())

    if not trains_dict:
        print('Length of circulation_data:', len(circulation_data))
        print('TravelData:', circulation_data[0]['TravelDate'])
        print('RameId', circulation_data[0]['RameId'])

async def update_train_positions(trains_dict: dict,
                                 last_gps_t: datetime,
                                 last_beacon_t: datetime,
                                 update_interval: int = 10) -> None:
    # Localisation

    while True:
        # Parallel fetch operations
        gps_updates, beacon_updates = await asyncio.gather(
            fetch_gps_update(last_gps_t),
            fetch_beacon_update(last_beacon_t)
            )


        # Pre-process beacon updates for efficient lookup
        beacon_lookup = {}
        for update in beacon_updates:
            for key in ['OTI_OperationalTrainNumber', 'ROTN_OTI_OperationalTrainNumber']:
                if key in update and update[key][-4:].isdigit():
                    num = int(update[key][-4:])
                    beacon_lookup[num] = update

        # Update the train objects
        for train in trains_dict.values():
            train_icomera_updates = [update for update in gps_updates if update['systemid'] in train.systemid]
            train_info_updates = [beacon_lookup[num] for num in train.train_number if num in beacon_lookup]
            train.update(train_icomera_updates, train_info_updates)
        
        if gps_updates:
            last_gps_t = gps_updates[-1]['timestamp']
        if beacon_updates:
            last_beacon_t = beacon_updates[-1]['LocationDateTime']

        # print('Positioning updating...for the following id', trains_dict.keys())
        # Sleep for some time before fetching updates again
        await asyncio.sleep(update_interval)


async def historical_train_positions(trains_dict: dict,
                                     start: datetime,
                                     end: datetime,
                                     rame_id: int) -> List:
    # Fetch historical train positions for a given date and train id, and send it to KF

    # to be processed by Kalman filter
    gps_raw, beacon_raw = await asyncio.gather(
            fetch_gps(trains_dict[rame_id].systemid, start, end),
            fetch_beacon(trains_dict[rame_id].train_number, start, end)
            )
    # Return the resutls of the KF
    return trains_dict[rame_id].update(gps_raw, beacon_raw)


async def get_realtime_train_data(trains_dict: dict, rame_id: int = None) -> List:
    # Return the latest attributes of train as a list of dictionary
    return [train.to_dict() for train in trains_dict.values() if rame_id is None or train.rame_id == rame_id]


async def get_realtime_train_data_geolocation(trains_dict: dict, rame_id: int = None) -> List:
    # Return the train data as a dictionary
    return [train.to_dict_geo() for train in trains_dict.values() if rame_id is None or train.rame_id == rame_id]
