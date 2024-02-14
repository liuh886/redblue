# train_service.py

from func.database import fetch_rame_info, fetch_primary_location, fetch_circulation_info, fetch_gps_update, fetch_beacon_update, fetch_gps, fetch_beacon
from fastapi import HTTPException
import asyncio
from func.kalman_filter import apply_kalman_filter
from typing import List
from datetime import datetime
from func.logger import StructuredLogger

class Train:
    def __init__(self,
                 rame_id: int,
                 rame_number: int  # trainset number
                 ):
        # Logger
        self.logger = None
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

    def update_service(self, circulation_record: dict):
        '''
        Update the service-related attributes of the train object with a new circulation record.

        Parameters:
        circulation_record: records from databse circulation table.

        '''
        train_number = int(circulation_record['TrainNumber'][-4:])
        um_train_number = int(
            circulation_record['UMTrainNumber'][-4:]) if circulation_record['UMTrainNumber'] != 'NULL' else None

        # Append new service to the train if it's not already listed
        if train_number not in self.train_number:
            self.train_number.append(train_number)
            self.traveldate.append(circulation_record['TravelDate'])
            self.origin.append(circulation_record['Origin'])
            self.destination.append(circulation_record['Destination'])
            self.status.append(circulation_record['Status'])
            self.type.append(circulation_record['Type'])
            self.um_train_number.append(
                um_train_number if um_train_number is not None else '')
            self.um_position.append(circulation_record['UMPosition'])
            self.total_distance.append(circulation_record['TotalDistance'])

    def update(self, new_gnss, new_beacons):
        '''
        Method to update the train object with data from GPS or Beacons
        '''

        # Handle multiple records. Sort updates by timestamp, and process from old to new

        # TODO: generate update matrix from new_gnss and new_beacons
        # print('length of new_gnss:', len(new_gnss))
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

        # to log the positioning events
        conditions_to_log = False

        if self.logger and conditions_to_log:
            self.update_logger()

        return results.to_dict()

    def update_logger(self):
        '''
        Update the logger to include positioning events
        '''

        self.logger.log('Train positioning events', extra={
            'latitude': self.latitute,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'speed': self.speed,
            'heading': self.heading,
            'lock': self.lock,
            'satellites': self.satellites,
            'quality': self.quality,
            'last_gps_t': self.last_gps_t,
            'last_beacon_t': self.last_beacon_t,
            'last_beacon_code': self.last_beacon_code,
        })

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

async def train_init(app) -> None:
    '''
    Initialize the app.trains_mapping_table with the Rame information from the database.
    '''

    rame_info = await fetch_rame_info()

    if not rame_info:
        app.logger.log(
            event="train_init_no_data",
            object_name="rame_info",
            object_type="database_table",
            content="No data found for the given Rames table",
            status="error",
            value="404",
            level="ERROR"
        )
        raise HTTPException(
            status_code=404, detail="No data found for the given Rames table")

    for item in rame_info:
        initialize_or_update_train(app, item)

    app.logger.log(
        event="train_init",
        object_name="trains_mapping_table",
        object_type="dictionary",
        content=f"Initialized Done",
        status="ok",
        value=None,
        level="INFO"
    )

def initialize_or_update_train(app, item):
    train_id = item['Id']
    rame_number = item['RameNumber']
    systemids = [int(str(rame_number) + '2'), int(str(rame_number) + '7')]  # TODO: Adjust for Noman rules

    if train_id not in app.trains_mapping_table:
        # Create a new Train object if not present in the mapping table
        train = Train(rame_id=train_id, rame_number=rame_number)
        train.systemid.extend(systemids)
        app.trains_mapping_table[train_id] = train
    else:
        # Update existing Train object
        train = app.trains_mapping_table[train_id]
        train.rame_id = train_id
        train.rame_number = rame_number
        train.systemid = systemids


async def beacon_init(app) -> None:
    '''
    Initialize the beacon table with the beacon information from the database.
    '''
    try:
        info = await fetch_primary_location(app.system_date, app.beacon_country)

        if not info:
            raise HTTPException(
                status_code=404, detail="No data found for the given beacon table")

        for i in info:
            app.update_beacon_mapping_table(i['Primary_Location_Code'],  {
                'country': i['Country_ISO_code'],
                'latitude': i['Latitude'],
                'longitude': i['Longitude']
            })
            
    except Exception as e:
        app.logger.log(
            event="beacon_init",
            object_type="core function",
            object_name="beacon_init",
            content=str(e),
            status="error",
            value=500,
            level="ERROR"
        )
        raise HTTPException(
            status_code=500, detail=f"Error fetching beacon data: {e}") from e


async def service_init(app) -> None:
    # Fetch the Circulation data for the given date
    circulation_data = await fetch_circulation_info(app.system_date)

    if not circulation_data:
        app.logger.log(
            event="service_init_no_data",
            object_name="circulation_data",
            object_type="database_table",
            content="No data found in Circulation table for the given date",
            status="error",
            value="404",
            level="ERROR"
        )
        raise HTTPException(
            status_code=404, detail="No data found in Circulation table for the given date")

    # Reset train services based on new circulation data
    for id in app.trains_dict.keys():
        train_obj = app.trains_dict[id]
        train_obj.reset_services()

    # Process each item in the circulation data
    for item in circulation_data:
        if item['RameId'] != 'NULL' and item['TrainNumber'][-4:].isdigit():
            # Valid records where RameId is not NULL
            id = int(item['RameId'])
            # Ensure the train object exists in trains_dict
            if id not in app.trains_dict and id in app.trains_mapping_table:
                app.trains_dict[id] = app.trains_mapping_table[id]
            # Update the service-related attributes of the train object
            if id in app.trains_dict:
                app.trains_dict[id].update_service(item)          
    # Log the initiation event
    app.log_event_initiation(circulation_data)
    
async def update_train_positions(app,
                                 update_interval: int = 10) -> None:
    '''
    Localisation update for all trains in app.trains_dict.
    '''

    while True:
        # Parallel fetch operations for GPS and Beacon updates
        gps_updates, beacon_updates = await asyncio.gather(
            fetch_gps_update(app.last_gps_t),
            fetch_beacon_update(app.last_beacon_t)
        )

        # Process GPS and Beacon updates
        process_updates(app, gps_updates, beacon_updates)

        # Update the last timestamps for GPS and Beacon data
        if gps_updates:
            app.last_gps_t = gps_updates[-1]['timestamp']
            app.length_gps = len(gps_updates)
        if beacon_updates:
            app.last_beacon_t = beacon_updates[-1]['LocationDateTime']
            app.length_beacon = len(beacon_updates)

        # Sleep before the next update cycle
        await asyncio.sleep(update_interval)

def process_updates(app, gps_updates, beacon_updates):
    beacon_lookup = {}

    # line by line to process the beacon updates
    for update in beacon_updates:
        
        # append coordinate to the update
        num, update_dict = get_beacon_update_info(update, app.beacon_mapping_table)
        
        # num is train number, using as a key, store the update in beacon_lookup
        if num is not None:
            beacon_lookup[num] = update_dict
        else:
            app.logger.log(
                event="create_beacon_lookup",
                object_name="beacon_update",
                object_type="BeaconUpdate",
                content="Invalid beacon update, missing or invalid operational train number",
                status="skip",
                value="",
                level="WARNING"
            )

    for train in app.trains_dict.values():
        # mapping two lists by systemid
        train_icomera_updates = [update for update in gps_updates if update['systemid'] in train.systemid]
        # mapping two lists by train number (num)
        train_info_updates = [beacon_lookup[num] for num in train.train_number if num in beacon_lookup]

        # Limit the data to the most recent records to avoid initial delays. To be removed in the future.
        gps_data = train_icomera_updates[-2000:] if len(train_icomera_updates) > 5000 else train_icomera_updates
        beacon_data = train_info_updates[-2000:] if len(train_info_updates) > 5000 else train_info_updates

        train.update(gps_data, beacon_data)

def get_beacon_update_info(update, beacon_mapping_table):
    '''
    Extract the operational train number from the beacon update and look up the corresponding beacon information.
    '''

    primary_num_key = 'OTI_OperationalTrainNumber'
    fallback_num_key = 'ROTN_OTI_OperationalTrainNumber'
    num = None
    update_dict = None

    for key in [primary_num_key, fallback_num_key]:
        if key in update and update[key][-4:].isdigit():
            num = int(update[key][-4:])
            primary_code = update.get('LocationPrimaryCode')
            beacon_info = beacon_mapping_table.get(int(primary_code))
            if beacon_info:
                update_dict = dict(update)
                update_dict['latitude'] = beacon_info['latitude']
                update_dict['longitude'] = beacon_info['longitude']
            break
    return num, update_dict


async def historical_train_positions(trains_dict: dict,
                                     start: datetime,
                                     end: datetime,
                                     rame_id: int,
                                     beacon_mapping_table: dict) -> List:
    '''
    Fetch historical train positions for a given date and train id, and send it to KF for localization.
    '''

    # to be processed by Kalman filter
    gps_raw, beacon_raw = await asyncio.gather(
            fetch_gps(trains_dict[rame_id].systemid, start, end),
            fetch_beacon(trains_dict[rame_id].train_number, start, end)
            )
    
    # mapping with beacon_mapping_table for coordinates
    for update in beacon_raw:
        primary_code = update.get('LocationPrimaryCode')
        if primary_code in beacon_mapping_table:
            beacon_info = beacon_mapping_table[primary_code]
            update_dict = dict(update)
            update_dict['latitude'] = beacon_info['latitude']
            update_dict['longitude'] = beacon_info['longitude']

    # Return the resutls of the KF
    return trains_dict[rame_id].update(gps_raw, beacon_raw)


async def get_realtime_train_data(trains_dict: dict, rame_id: int = None) -> List:
    # Return the latest attributes of train as a list of dictionary
    return [train.to_dict() for train in trains_dict.values() if rame_id is None or train.rame_id == rame_id]


async def get_realtime_train_data_geolocation(trains_dict: dict, rame_id: int = None) -> List:
    # Return the train data as a dictionary
    return [train.to_dict_geo() for train in trains_dict.values() if rame_id is None or train.rame_id == rame_id]
