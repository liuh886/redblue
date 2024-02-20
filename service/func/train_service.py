# train_service.py

from func.database import fetch_rame_info, fetch_primary_location, fetch_circulation_info, fetch_gps_update, fetch_beacon_update, fetch_gps, fetch_beacon
from fastapi import HTTPException
import asyncio
from func.kalman_filter import apply_kalman_filter
from typing import List, Union
from datetime import datetime
from func.system import SystemStatus
from kalman.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from kalman.common import Q_continuous_white_noise
from kalman.common import Q_discrete_white_noise
import numpy as np
import pandas as pd
import utm
import time

    
class Train:
    '''
    This is a train classs equiped with GPS and beacons sensors and kalman filter.
    
    Definition of terms:
        - The reference point: the front of the train.
        - The offset: the distance from the reference point to the front and back of the GPS.
        - The orientation (theta): the direction of the train (relative to North). The azimuth angle of the vector of physical tail to head.
        - The heading (phi): the direction of moving (relative to North), baed on the GPS on head. The azimuth angle of the vector from previous's ref position to next position.
        - The inverted: the train's orientation is opposite to the heading.
        - The KF and control theory terms:
            - Update: the process of updating the train's state by the sensor's signals.
            - Prediction: the process of calculating the train's state in the future (dt). 
            - x: the state vectors.
            - zs: the measurement vectors.
            - Hs: the measurement matrices. Turning the state space into the measurement space.
            - Rs: the measurement noise matrices.
            - Fs: the state transition matrices.
            - Qs: the process noise matrices.
    '''

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
        self.utm_zone = None
        self.gps_sys = 'ICOMERA'
        # service.
        self.init_services()
        # positioning signal status. This is the most recent signals.
        self.latitute = None
        self.longitude = None
        self.east = None
        self.north = None
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
        points = MerweScaledSigmaPoints(n=12, alpha=0.1, beta=2., kappa=0)  # Adjust these parameters as needed
        self.kf = UnscentedKalmanFilter(dim_x=12, dim_z=4, dt=1, fx=self.kf_fx, hx=self.kf_hx_gps_front, points=points)
        self.previous_timestamp = pd.Timestamp('1900-01-01', tz='UTC')
        self.orientation_kf = 0  # [deg] the vectors azimuth agle of physical tail to head. 

        # Kalman filter initial state
        if self.east:
            self.kf.x = np.array([self.east,    # east
                                  self.north,   # north
                                  0,            # east of second GPS
                                  0,            # north of second GPS
                                  self.speed,            # velocity [m/s]
                                  0,            # acc [m/s^2]
                                  0,            # heading front gps [deg]
                                  0,            # rate of turning front GPS[deg/s]
                                  0,            # heading back gps [deg]
                                  0,            # rate of turning back gps [deg/s]
                                  0,            # orientation [deg]
                                  0,            # lag [s]
                                  ])
        else:
            self.kf.x = np.array([0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.])

    def kf_fx(self, x, dt):
        '''
        State transition matrix/function for the Kalman filter.

        Core to kf.x

        This is a nonlinear process, so we use the Unscented Kalman Filter (UKF) to handle it.
        '''

        # Assuming a simple constant velocity model for demonstration
        # Extract state components for readability
        x_east, y_north, x_east_2, y_north_2, velocity, acc, heading_1, rate_of_turn_1, heading_2, rate_of_turn_2, orientation, lag = x

        # Update positions based on velocity and heading
        # For the front GPS (GPS 1)
        dx_1 = (velocity + 0.5 * acc * dt) * np.cos(np.radians(heading_1)) * dt
        dy_1 = (velocity + 0.5 * acc * dt ) * np.sin(np.radians(heading_1)) * dt

        # For the rear GPS (GPS 2), assuming it follows the same velocity but might have a different heading
        dx_2 = (velocity + 0.5 * acc * dt) * np.cos(np.radians(heading_2)) * dt
        dy_2 = (velocity + 0.5 * acc * dt) * np.sin(np.radians(heading_2)) * dt

        # Update velocity based on acceleration
        new_velocity = velocity + acc * dt

        # Update headings based on rate of turn
        new_heading_1 = heading_1 + rate_of_turn_1 * dt
        new_heading_2 = heading_2 + rate_of_turn_2 * dt
        
        # Update orientation|azimuth angle of the vector of physical tail to head
        new_orientation = np.degrees(np.arctan2((y_north_2 - y_north - dy_1 + dy_2), (x_east_2 - x_east - dx_1 + dx_2)))
        
        # Update lag
        new_lag = self.length / (velocity + 0.5 * acc * dt) 

        # Update the state vector with new values
        x_updated = x.copy()
        x_updated[0] += dx_1  # Update east position of GPS 1
        x_updated[1] += dy_1  # Update north position of GPS 1
        x_updated[2] += dx_2  # Update east position of GPS 2
        x_updated[3] += dy_2  # Update north position of GPS 2
        x_updated[4] = new_velocity  # Update velocity
        x_updated[6] = new_heading_1  # Update heading for GPS 1
        x_updated[8] = new_heading_2  # Update heading for GPS 2
        x_updated[10] = new_orientation  # Update orientation
        x_updated[11] = new_lag  # Update lag

        return x_updated
    
    def kf_hx_gps_front(self, x):
        '''
        Measurement matrix/function for the Kalman filter. Turning the state space into the measurement space.

        Core to kf.x
        '''
        # Unpack the state vector for readability
        x_east, y_north, x_east_2, y_north_2, velocity, acc, heading_1, rate_of_turn_1, heading_2, rate_of_turn_2, orientation, lag = x

        # For GPS 2 (Rear)
        # This is not the optimal way to do this, because (1) it assume the train is moving straight.
        # (2) the current GPS_front is not the neastest update.
        z_east_2 = x_east  - lag * (velocity - 0.5 * acc * lag) * np.cos(np.radians(heading_1))
        z_north_2 = y_north - lag *  (velocity - 0.5 * acc * lag) * np.sin(np.radians(heading_1))
        z_heading_2 = heading_1 - lag *  (rate_of_turn_1)

        return [x_east, y_north, z_east_2, z_north_2, velocity, heading_1, z_heading_2]
    
    def kf_hx_gps_back(self, x):

        # Unpack the state vector for readability
        x_east, y_north, x_east_2, y_north_2, velocity, acc, heading_1, rate_of_turn_1, heading_2, rate_of_turn_2, orientation, lag = x

        return [x_east_2, y_north_2, velocity heading_2]
    
    def kf_hx_beacon(self, x):
        return [x[0], x[1]]
    
    def kf_Q(self,dt):
        '''
        Process noise matrix for the Kalman filter.

        Of of the most difficult part of the Kalman filter is to determine the process noise matrix Q.

        Determining the spectral density / var is an engineering and experientally tune processes. 
        (continuous white modeel/discrete).

        If dt is very small, setting it to zero except for a noise term in the lower rightmost element.

        Parameters:
        dim: 
        blok_size: 2 becasue two dimentional system x,y 
        order_by_dim=False beacsue x, y, z, vx, vy, vz not x, vx, y, vy, z, vz

        '''
        return Q_discrete_white_noise(dim=12, dt=dt, var=1, block_size=2, order_by_dim=False)
        
    def kf_orientation(self, condition: bool = False):
        '''
        Orientation of the train.

        True for forward, False for backward.
        '''

        if self.orientation_kf is None:
            self.orientation_kf = True
        
        # Toggle the orientation when
        if condition:
            self.orientation_kf = not self.orientation_kf

        return self.orientation_kf

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
        self.orientation = []  # 
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
            self.orientation.append(circulation_record['Positioning'])
            self.um_train_number.append(
                um_train_number if um_train_number is not None else '')
            self.um_position.append(circulation_record['UMPosition'])
            self.total_distance.append(circulation_record['TotalDistance'])

    def transform_to_matrix(self, new_gnss: pd.DataFrame, new_beacons: pd.DataFrame):

        """
        Transforms GNSS and beacon data into matrices suitable for the Kalman Filter batch process.
        Returns:
            - measurement vectors (zs), 
            - observation matrices (Hs), 
            - measurement noise matrices (Rs),
            - state transition matrices (Fs).
            - process noise matrices (Qs),
            - and meta data dataframe.
        """

        combined_data = []

        # TODO: Adjust the values according to HDOP, VDOP, PDOP, and TDOP
        R_gps = np.diag([5**2, 5**2, (1/3.6)**2, (1/3.6)**2])  # Assuming 5m position error, 1 m/s speed error (3.6 KMH)
        R_beacon = np.diag([10**2, 10**2])  # Assuming 10m position error for beacon

        # Process GNSS data
        if new_gnss is not None and not new_gnss.empty:
            # (1) Process GNSS data into DataFrame
            new_gnss['sensor_type'] = 'GNSS'
            new_gnss['sensor_id'] = new_gnss['systemid']
            new_gnss['H'] = [self.kf_hx_gps_front if new_gnss['sensor_id'][-1]==2 else self.kf_hx_gps_back] * len(new_gnss)
            new_gnss['R'] = [R_gps] * len(new_gnss)
            new_gnss['z'] = np.array([new_gnss['east'], 
                              new_gnss['north'], 
                              new_gnss['speed'] * np.cos(np.deg2rad(new_gnss['heading'])), 
                              new_gnss['speed'] * np.sin(np.deg2rad(new_gnss['heading']))])
            combined_data.append(new_gnss[['timestamp', 'sensor_type', 'sensor_id', 'H', 'R', 'z']])

        # Process Beacon data
        if new_beacons is not None and not new_beacons.empty:
            new_beacons['sensor_type'] = 'Beacon'
            new_beacons['sensor_id'] = new_beacons['primary_location_code']
            new_beacons['H'] = [self.kf_hx_beacon] * len(new_beacons)
            new_beacons['R'] = [R_beacon] * len(new_beacons)
            new_beacons['z'] = np.array([new_beacons['east'], new_beacons['north']])
            combined_data.append(new_beacons[['timestamp','sensor_type', 'sensor_id', 'H', 'R', 'z']])

        # Merge and sort combined data
        combined_df = pd.concat(combined_data).sort_values(by='timestamp')

        # Calculate dt and filter out delayed updates
        # Drop signlas comes in delayed
        process_df = combined_df[combined_df['timestamp'] > self.previous_timestamp]
        init_dt = process_df['timestamp'].iloc[0] - self.previous_timestamp  # Initial dt of this batch
        process_df['dt'] = process_df['timestamp'].diff().dt.total_seconds().fillna(init_dt)

        # Construct Fs based on dt
        zs = process_df['z'].tolist()
        Hs = process_df['H'].tolist()
        Rs = process_df['R'].tolist()
        Fs = self.kf_fx
        Qs = [self.kf_Q(dt) for dt in process_df['dt']]

        metadata = process_df[['timestamp', 'sensor_type', 'sensor_id', 'dt', 'z']]

        # Update previous_timestamp
        if not process_df.empty:
            self.previous_timestamp = process_df.iloc[-1]['timestamp']

        # update logger
        if self.logger:
            if len(combined_df) > len(process_df):
                unprocessed = combined_df[combined_df['timestamp'] <= self.previous_timestamp]

                self.logger.log(
                    event="Signal_delayed",
                    object_name="Klamn filter batch process",
                    object_type="signal updates",
                    content=f"Signal comes in delay, are dropped from the batch process:\n"
                    f"{unprocessed['timestamp', 'sensor_type', 'sensor_id']}",
                    status="Skiped",
                    value=len(unprocessed),
                    level="INFO"
                )

        return np.array(zs), np.array(Hs), np.array(Rs), np.array(Fs), np.array(Qs), metadata

    def transform_to_df(self, mu, cov, metadata: pd.DataFrame):
        '''
        Transform the Kalman filter output into two DataFrames:
        1. A DataFrame with latitude, longitude, heading, and speed along with their covariances and metadata.
        2. An extended DataFrame that includes all states and covariances from the Kalman filter output along with the metadata.
        '''

        # Extract the means for east, north, v_east, and v_north
        east = mu[:, 0].flatten()
        north = mu[:, 1].flatten()
        v_east = mu[:, 2].flatten()
        v_north = mu[:, 3].flatten()
        
        # Convert UTM coordinates back to lat/lon for each row
        lat_lon = [utm.to_latlon(e, n, zone_number=self.utm_zone) for e, n in zip(east, north)]
        latitude, longitude = zip(*lat_lon)

        # Calculate heading and speed from v_east and v_north
        heading = np.degrees(np.arctan2(v_north, v_east)) % 360
        speed = np.sqrt(v_east**2 + v_north**2) * 3.6  # Convert from m/s to km/h

        # Create a simplified DataFrame for results
        results = pd.DataFrame({
            'latitude': latitude,
            'longitude': longitude,
            'east': east,
            'north': north,
            'speed': speed,
            'heading': heading
        })

        # Add covariance for latitude, longitude, speed, heading
        results['cov_latitude'] = cov[:, 0, 0]  # Proxy for latitude covariance
        results['cov_longitude'] = cov[:, 1, 1]  # Proxy for longitude covariance
        results['cov_speed'] = (cov[:, 2, 2] + cov[:, 3, 3] ) / 2  # Average of speed covariance

        # Merge simplified results with metadata
        metadata_subset = metadata[['timestamp', 'sensor_type', 'sensor_id', 'dt']]
        results = pd.concat([metadata_subset.reset_index(drop=True), results.reset_index(drop=True)], axis=1)

        # Merge extended results with metadata
        extended_results = metadata[['mu', 'cov']]= [mu, cov]

        return results, extended_results

    
    def update(self, new_gnss: Union[pd.DataFrame, None], new_beacons: Union[pd.DataFrame, None]):
        '''
        Method to update the train object with data from GPS or Beacons.
        '''

        # turn new_gnss and into new_beacons zs matrix, Hs matrix, Rs matrix in ASC order by timestamp
        zs, Hs, Rs, Fs, Qs, df_meta = self.transform_to_matrix(new_gnss, new_beacons)

        # Kalman filter return a matrix of states, and covariance matrix
        mu, cov = self.kf.batch_filter(zs,Fs=Fs, Rs=Rs, dts= df_meta['dt'], Qs=Qs, Hs=Hs)
    
        # transform the matrix of retust into a dataframe
        results, extended_results = self.transform_to_df(mu, cov, df_meta)

        # Only update the train with the last position from the Kalman filter
        # .item() is to convert the numpy types to native Python types that are JSON serializable.
        if not results.empty:
            self.latitute = results['filtered'].iloc[-1].item()
            self.longitude = results['filtered'].iloc[-1].item()
            self.east = results['east'].iloc[-1].item()
            self.north = results['north'].iloc[-1].item()
            #self.altitude = results['filtered'].iloc[-1].item()
            self.speed = results['speed'].iloc[-1].item()
            self.heading = results['heading'].iloc[-1].item()
        
        # Update attributes reltated to sensor's as needed
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
            self.last_beacon_code = new_beacons.index.get_level_values('primary_location_code')[-1]
            self.last_beacon_t_against_booked = new_beacons.iloc[-1]['against_booked']

        return results.to_dict(orient='records'), extended_results


    def to_dict(self):
        # Define a dictionary to hold serializable attributes
        serializable_dict = {}
        
        # List of attributes to be included in the dict
        attrs_to_include = ['rame_id', 'rame_number', 'systemid', 'train_number', 'traveldate', 
                            'origin', 'destination', 'status', 'type','inverted' ,'um_train_number', 
                            'um_position', 'last_gps_t', 'last_beacon_t', 'last_beacon_code', 
                            'lock', 'satellites', 'quality', 'latitute', 'longitude', 
                            #'altitude', 
                            'speed', 'heading']

        # Iterate over specified attributes
        for attr in attrs_to_include:
            attr_value = getattr(self, attr, None)

            # Convert numpy types to Python native types
            if isinstance(attr_value, np.generic):
                serializable_dict[attr] = attr_value.item()
            else:
                serializable_dict[attr] = attr_value
        
        return serializable_dict

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
        train.utm_zone = sys.utm_zone
    else:
        # If already in table , update the existing Train object
        train = sys.trains_mapping_dict[train_id]
        train.rame_id = train_id
        train.rame_number = rame_number
        train.systemid = systemids
        train.utm_zone = sys.utm_zone

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
        # Set the index of the DataFrame
        df['primary_location_code'] = df['primary_location_code'].astype(str)
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
    try:
        for train in sys.trains_dict.values():
            gps_data = gps_df[gps_df['systemid'].isin(train.systemid)] if not gps_df.empty else None
            beacon_data = beacon_df[beacon_df['train_number'].isin(train.train_number)] if not beacon_df.empty else None

            train.update(gps_data, beacon_data)

    except KeyError as e:
        # Log the error with more details
        print(f"KeyError occurred: {e}")
        print("DataFrame before error:")
        print(beacon_data.head())  # Adjust this line as needed to log more or less data
        raise e

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
            'MessageDateTime': 'message_datetime',
            'AgainstBooked': 'against_booked'
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
    data, _ = trains_dict[rame_id].update(gps_df, beacon_df)

    return data

async def get_realtime_train_data(trains_dict: dict, rame_id: int = None) -> List: # type: ignore
    # Return the latest attributes of train as a list of dictionary
    return [train.to_dict() for train in trains_dict.values() if rame_id is None or train.rame_id == rame_id]


async def get_realtime_train_data_geolocation(trains_dict: dict, rame_id: int = None) -> List: # type: ignore
    # Return the train data as a dictionary
    return [train.to_dict_geo() for train in trains_dict.values() if rame_id is None or train.rame_id == rame_id]
