# train_service.py

from func.database import fetch_rame_info, fetch_primary_location, fetch_circulation_info, fetch_gps_update, fetch_beacon_update, fetch_gps, fetch_beacon
from fastapi import HTTPException
import asyncio
from func.kalman_filter import apply_kalman_filter
from typing import List, Union
from datetime import datetime
from func.system import SystemStatus
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.common import Q_continuous_white_noise
from filterpy.common import Q_discrete_white_noise
import numpy as np
import pandas as pd
import utm
import time

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
        # service.
        self.init_services()
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
        # initial Kalman filter
        self.init_kf()

    def init_kf(self):
        '''
        Initialize the Kalman filter for the train object.
        '''
        # Kalman filter.
        self.kf = KalmanFilter(dim_x=6, dim_z=2)

        # Kalman filter initial state
        if self.latitute:
            self.kf.x = np.array([self.latitute, 
                                  self.longitude, 
                                  self.altitude, 
                                  self.speed, 
                                  self.heading,
                                  0])
        else:
            self.kf.x = np.array([0., 0., 0., 0., 0., 0.])  # initial state (location, velocity, heading, acc)

        self.kf.F = np.array([[1., 0., 1., 0., 0.],
                              [0., 1., 0., 1., 0.],
                              [0., 0., 1., 0., 1.],
                              [0., 0., 0., 1., 0.],
                              [0., 0., 0., 0., 1.]])  # state transition matrix
        
        self.kf.H = np.array([[1., 0., 0., 0., 0.],
                                [0., 1., 0., 0., 0.]])
        self.kf.P *= 1000.  # covariance matrix
        self.kf.R = 5  # measurement uncertainty
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=1., var=0.13)  # process uncertainty
        
    def init_services(self):
        '''
        Reset the service-related attributes of the train object.

        A trainset can have multiple train service in a day. Thus the list.
        '''
        self.train_number = []  # int
        self.traveldate = []  # datetime
        self.origin = []  # str
        self.destination = [] # str
        self.status = []  # str
        self.type = []  # str
        self.invert = []  # 
        self.um_train_number = []   # int
        self.um_position = []   # str
        self.total_distance = []  # int

    def update_service(self, circulation_record: dict):
        '''
        Update the service-related attributes of the train object with a new circulation record.

        Parameters:
        circulation_record: records from databse's circulation table.

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

    def update(self, new_gnss: Union[pd.DataFrame, None], new_beacons: Union[pd.DataFrame, None]):
        '''
        Method to update the train object with data from GPS or Beacons.

        How many records come, how many records to process, to export.
        '''

        # Handle multiple records. Sort updates by timestamp, and process from old to new

        # TODO: generate update matrix from new_gnss and new_beacons
        # print('length of new_gnss:', len(new_gnss))
        # the last states matrix of train
        
        matrix_old = [[self.latitute, self.longitude, self.altitude, self.speed, self.heading]]

        # Kalman filter return a matrix of states
        results = apply_kalman_filter(matrix_old)

        # Update attributes as needed
        if new_gnss is not None and not new_gnss.empty:
            # GNSS metadata
            self.last_gps_code = new_gnss.iloc[-1]['systemid']
            self.lock = new_gnss.iloc[-1]['lock']
            self.satellites = new_gnss.iloc[-1]['satellites']
            self.quality = new_gnss.iloc[-1]['quality']

            # timestamp metadata
            self.last_gps_t = new_gnss.iloc[-1]['timestamp']

        if new_beacons is not None and not new_beacons.empty:
            self.last_beacon_t = new_beacons.iloc[-1]['timestamp']
            self.last_beacon_code = new_beacons.iloc[-1]['primary_location_code']

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
        # TODO: adjust. No need to log all the attributes.
        #self.logger.log('Train positioning events')

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
            'kalman_filter_state': self.kf.x_prior,
            'kalman_filter_gain': None,
            'kalman_filter_prediction': self.kf.x_post,
            'kalman_filter_residual': self.kf._likelihood,
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

async def train_init(sys: SystemStatus) -> None:
    '''
    Initialize the app.trains_mapping_dict with the Rame information from the database.

    Parameters:
    app: the environemnt of train instance, FastAPI application instance.

    Operations:
    Fetch the Rame information from the database, and use it update app.trains_mapping_dict
    '''

    rame_info = await fetch_rame_info()

    if not rame_info:
        sys.logger.log(
            event="train_init_no_data",
            object_name="rame_info",
            object_type="database_table",
            content="No data found for the given Rames table",
            status="error",
            value="404",
            level="ERROR"
        ) if sys.logger else None
        raise HTTPException(
            status_code=404, detail="No data found for the given Rames table")

    for item in rame_info:
        initialize_or_update_train(sys, item)

    sys.logger.log(
        event="train_init",
        object_name="trains_mapping_dict",
        object_type="dictionary",
        content=f"Initialized Done",
        status="ok",
        value=None,
        level="INFO"
    ) if sys.logger else None

def initialize_or_update_train(sys, item):
    '''
    Creating train objects based on database rame table
    '''

    # info from table. All turn into int type.
    train_id = int(item['Id'])
    rame_number = int(item['RameNumber'])
    systemids = [int(str(rame_number) + '2'), int(str(rame_number) + '7')]  # TODO: Adjust for Noman rules

    if train_id not in sys.trains_mapping_dict:
        # Create a new Train object if not present in the mapping table
        train = Train(rame_id=train_id, rame_number=rame_number)
        train.systemid.extend(systemids) # type: ignore
        # Add it to the mapping table
        sys.trains_mapping_dict[train_id] = train
    else:
        # If already in table , update the existing Train object
        train = sys.trains_mapping_dict[train_id]
        train.rame_id = train_id
        train.rame_number = rame_number
        train.systemid = systemids

async def beacon_init(sys) -> None:
    '''
    Initialize the beacon table with the beacon information from the database.

    Operations:
    Fetch the beacon information from the database, and use it update app.beacon_mapping_df
    '''
    try:
        info = await fetch_primary_location(sys.system_date, sys.beacon_country)

        if not info:
            raise HTTPException(status_code=404, detail="No data found for the given beacon table")

        # Turning beacon table into a dataframe, mapping with NE coordinates
        df = pd.DataFrame(info)
        # rename the columns
        df = df.rename(columns={'Country_ISO_code': 'country_code',
                                'Primary_Location_Code': 'primary_location_code',
                                'Latitude': 'latitude',
                                'Longitude': 'longitude',
                                'Start_Validity': 'start_validity'
                            })
        # Drop the rows with missing values
        df.dropna(subset=['latitude', 'longitude','primary_location_code','country_code'], inplace=True)

        # Add UTM coordinates to the DataFrame
        df['east'], df['north'], _, _ = utm.from_latlon(df['latitude'].values,  # type: ignore
                                                df['longitude'].values,
                                                force_zone_number=sys.utm_zone)
        
        df.set_index(['country_code', 'primary_location_code'], inplace=True)
        sys.beacon_mapping_df = df

    except Exception as e:
        sys.logger.log(
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


async def service_init(sys) -> None:
    # Fetch the Circulation data for the given date
    circulation_data = await fetch_circulation_info(sys.system_date)

    if not circulation_data:
        sys.logger.log(
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
    else:
        # Reset train services when there is new circulation data
        for train in sys.trains_dict.values():
            train.init_services()

    # Process each item in the circulation data
    for item in circulation_data:
        if item['RameId'] != 'NULL' and item['TrainNumber'][-4:].isdigit():
            # Valid records where RameId is not NULL
            id = int(item['RameId'])

            if id in sys.trains_dict:
                sys.trains_dict[id].update_service(item)
            # Update the service-related attributes of the train object
            elif id not in sys.trains_dict and id in sys.trains_mapping_dict:
                sys.trains_dict[id] = sys.trains_mapping_dict[id]
            else:
                sys.logger.log(
                    event="service_init",
                    object_name="trains_dict, trains_mapping_dict",
                    object_type="dictionary",
                    content=f"No data found in trains_mapping_dict table for the give RameId: {id}",
                    status="skip",
                    value="404",
                    level="ERROR"
                )
                raise HTTPException(
                    status_code=404, detail="No data found in trains_mapping_dict table for the give RameId: {id}")

    # Log the initiation event
    sys.log_event_initiation(circulation_data)
    
async def fetch_train_positions(sys,
                                update_interval: int = 10) -> None:
    '''
    Localisation update for all trains in app.trains_dict.
    '''

    while len(sys.trains_dict) != 0:
        # Parallel fetch operations for GPS and Beacon updates
        gps_updates, beacon_updates = await asyncio.gather(
            fetch_gps_update(sys.last_gps_t, sys.end_timestamp),
            fetch_beacon_update(sys.last_beacon_t, sys.end_timestamp)
        )
        
        # TODO : Delete the print and timing
        t_2 = time.time()
        print(f"-----Application - KF tracker - The querys takes {t_2 - sys.t_1:.2f} seconds")

        # Process GPS and Beacon updates
        apply_updates(sys, gps_updates, beacon_updates)

        print(f"-----Application - KF tracker - The calculation takes {time.time() - t_2:.2f} seconds."
              f" Now the query datetime: {sys.last_gps_t}.")
        
        # Sleep before the next update cycle
        await asyncio.sleep(update_interval)

        sys.t_1 = time.time()

    raise HTTPException(status_code=404, detail="No train found in the system, please re-initiate the system with a new date.")


def apply_updates(sys, gps_updates, beacon_updates):
    # Initialize empty DataFrames in case there are no updates
    gps_df = pd.DataFrame()
    beacon_df = pd.DataFrame()

    # Update the last timestamps for GPS and Beacon data
    if gps_updates:
        sys.last_gps_t = gps_updates[-1]['timestamp']
        sys.length_gps = len(gps_updates)
        gps_df = create_gps_updates_df(gps_updates)

    if beacon_updates:
        sys.last_beacon_t = beacon_updates[-1]['LocationDateTime']
        sys.length_beacon = len(beacon_updates)
        beacon_df = create_beacon_updates_df(beacon_updates, sys.beacon_mapping_df)


    # Update qurey window
    sys.end_timestamp = max(sys.last_gps_t, sys.last_beacon_t) + sys.query_step
        
    # (3) send update to trains
    for train in sys.trains_dict.values():
        gps_data = gps_df[gps_df['systemid'].isin(train.systemid)] if not gps_df.empty else None
        beacon_data = beacon_df[beacon_df['train_number'].isin(train.train_number)] if not beacon_df.empty else None

        train.update(gps_data, beacon_data)

def create_beacon_updates_df(beacon_updates, beacon_mapping_df = None):
    '''
    Create a DataFrame for beacon updates.
    '''
    try:
        df = pd.DataFrame(beacon_updates)
        df = df.rename(columns={
            'OTI_OperationalTrainNumber': 'OTI_train_number',
            'ROTN_OTI_OperationalTrainNumber': 'ROTN_OTI_train_number',
            'LocationPrimaryCode': 'primary_location_code',
            'LocationDateTime': 'timestamp',
            'CountryCodeISO': 'country_code',
            'MessageDateTime': 'message_datetime'
        })
        df['OTI_train_number'] = df["OTI_train_number"].fillna(df["ROTN_OTI_train_number"])
        df['train_number'] = pd.to_numeric(df['OTI_train_number'].str[-4:], errors='coerce')
        
        df.set_index(['country_code', 'primary_location_code'], inplace=True)
     
    except KeyError as e:
        # Log the error with more details
        print(f"KeyError occurred: {e}")
        print("DataFrame before error:")
        print(df.head())  # Adjust this line as needed to log more or less data
        raise e
    if beacon_mapping_df is not None:
        # Perform index-based joining to add coordinates to the beacon updates
        df = df.join(beacon_mapping_df, how='left')

    return df

def create_gps_updates_df(gps_updates, sys = None):
    '''
    Create a DataFrame for GPS updates.
    '''

    df = pd.DataFrame(gps_updates)
    df = df.rename(columns={
        'systemid': 'systemid',
        'latitude': 'latitude',
        'longitude': 'longitude',
        'altitude': 'altitude',
        'speed': 'speed',
        'heading': 'heading',
        'metaData.lock': 'lock',
        'metaData.satellites': 'satellites',
        'quality': 'quality',
        'timestamp': 'timestamp'
    })
    if sys is not None:
        df['east'], df['north'], _, _ = utm.from_latlon(df['latitude'].values,
                                        df['longitude'].values,
                                        force_zone_number=sys.utm_zone)
    
    return df


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
    
    print('length of gps_raw:', len(gps_raw))
    print('length of beacon_raw:', len(beacon_raw))

    gps_df = create_gps_updates_df(gps_raw) if gps_raw else None
    beacon_df = create_beacon_updates_df(beacon_raw, beacon_mapping_table) if beacon_raw else None
    
    # Return the resutls of the KF
    return trains_dict[rame_id].update(gps_df, beacon_df)


async def get_realtime_train_data(trains_dict: dict, rame_id: int = None) -> List: # type: ignore
    # Return the latest attributes of train as a list of dictionary
    return [train.to_dict() for train in trains_dict.values() if rame_id is None or train.rame_id == rame_id]


async def get_realtime_train_data_geolocation(trains_dict: dict, rame_id: int = None) -> List: # type: ignore
    # Return the train data as a dictionary
    return [train.to_dict_geo() for train in trains_dict.values() if rame_id is None or train.rame_id == rame_id]
