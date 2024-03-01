# train_service.py

from func.database import fetch_rame_info, fetch_primary_location, fetch_circulation_info, fetch_gps_update, fetch_beacon_update, fetch_gps, fetch_beacon
from fastapi import HTTPException
import asyncio
from typing import List, Union
from datetime import datetime
from func.system import SystemStatus
from kalman.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from kalman.common import Q_discrete_white_noise_two_gps
import numpy as np
import pandas as pd
import utm
import time
from pyproj import Transformer
import scipy
from scipy.stats import circmean
import numpy as np

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
        self.length = 100   # the geometric length of the two GPS
        self.rame_number = rame_number  # trainset number
        self.systemid = []
        self.to_en : Transformer = None
        self.to_lonlat : Transformer =  None
        self.gps_sys = 'ICOMERA'
        # service.
        self.init_services()
        # positioning signal status. This is the most recent signals.
        self.latitute_h = None
        self.longitude_h = None
        self.east_h = None
        self.north_h = None
        self.altitude = None
        self.speed = None 
        self.heading_h = None  # GPS1 - head
        self.heading_t = None  # GPS2 - tail
        self.lock = None
        self.satellites = None
        self.quality = None
        self.last_gps_t = None
        self.last_gps_code = None  # ICOMERA/Noman system id
        self.last_beacon_t = None
        self.last_beacon_code = None
        # timestamp of the last update
        self.previous_timestamp = pd.Timestamp('1900-01-01', tz='UTC')
        self.orientation_N72 = None
        self.orientation_N27 = None
        # initial Kalman filter
        self.init_kf()

    def init_kf(self):
        '''
        Initialize the Unsented Kalman filter for the train object.

        SigmaPoints:
            alpha: 0.001
                If the filter is too optimistic (underestimating uncertainty), 
                increasing alpha can help spread the sigma points further from the mean, capturing more of the uncertainty. 
                However, if the filter is too erratic, decreasing alpha might help.
            beta = 2.
                If you have specific knowledge about the distribution of your state 
                (e.g., heavy-tailed or skewed), adjusting beta can help improve performance. 
                Otherwise, keeping it at 2 is a reasonable default for Gaussian-like distributions.
            Kappa = 0

        Some diffeerent opinions on paramters http://mlg.eng.cam.ac.uk/pub/pdf/TurRas10.pdf

        '''
        # Kalman filter.

        def custom_x_mean_fn(sigmas, Wm):
            '''
            Define a custom mean function to handle the angles in x correctly.
            '''
            x = np.zeros(12)  # Adjust for your state vector length

            for i in range(len(sigmas)):
                s = sigmas[i]
                # Handle linear components directly
                for j in range(12):
                    if j not in [6, 8]:  # Skip the 6th and 8th elements (angles)
                        x[j] += s[j] * Wm[i]

            # Compute circular mean for angles
            angle_indices = [6, 8]
            for angle_index in angle_indices:
                angles = sigmas[:, angle_index]
                circular_mean = circmean(angles, high=np.pi, low=-np.pi)
                x[angle_index] = circular_mean

            return x

        def custom_z_mean_fn(sigmas, Wm):
            '''
            Define a custom mean function to handle the angles in z correctly.
            '''
            z = np.zeros(sigmas.shape[1])  # Adjust for the width of sigmas
            angle_indices = [3]  # Indices of the angle elements in z
            linear_indices = [i for i in range(sigmas.shape[1]) if i not in angle_indices]

            # Handle linear components
            for i in range(len(sigmas)):
                s = sigmas[i]
                for j in linear_indices:
                    z[j] += s[j] * Wm[i]
                    
            # When there is angle measurements
            if sigmas.shape[1] == 4:
                for angle_index in angle_indices:
                    angles = sigmas[:, angle_index]
                    circular_mean = circmean(angles, high=np.pi, low=-np.pi)
                    z[angle_index] = circular_mean

            return z

        def residual_x(x, U):
            '''
            Define a custom residual function to handle the angles in x correctly.
            Ensure that inputs are real numbers to avoid complex number operations.
            '''
            # Ensure inputs are real; otherwise, take the real part
            if not np.isrealobj(x) or not np.isrealobj(U):
                x = np.real(x)
                U = np.real(U)

            y = x - U

            # Use np.angle to ensure the angle is within -pi to pi range
            y[6] = np.angle(np.exp(1j * y[6]))  # For heading 1
            y[8] = np.angle(np.exp(1j * y[8]))  # For heading 2
            return y

        def residual_z(a, b):
            '''
            Define a custom residual function to handle the angles in z correctly.
            Ensure that inputs are real numbers to avoid complex number operations.
            '''
            y = a - b
            if len(y) >= 4:
                y[3] = np.angle(np.exp(1j * y[3]))  # For angle measurement
            return y

        def get_near_psd(P, max_iter=100):
            '''
            Get the nearest positive semi-definite matrix.
            https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite            
            The equations are from
            N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
            '''
            # adding a jitter.
            eps = 1e-3  # Small positive value for regularization
            increment_factor = 10  # Factor to increase eps if needed

            def is_symmetric(A):
                return np.allclose(A, A.T)
            
            def is_positive_definite(A):
                try:
                    np.linalg.cholesky(A)
                    return True
                except np.linalg.LinAlgError:
                    return False

            for _ in range(max_iter):
                if is_symmetric(P) and is_positive_definite(P):
                    return P  # The matrix is already suitable for Cholesky

                # Make P symmetric
                P = (P + P.T) / 2

                # Set negative eigenvalues to zero and ensure strict positivity
                eigval, eigvec = np.linalg.eig(P)
                eigval[eigval < 0] = 0
                eigval += eps  # Ensure all eigenvalues are strictly positive

                # Reconstruct the matrix
                P = eigvec.dot(np.diag(eigval)).dot(eigvec.T)

                # Check if P is now positive definite
                if is_positive_definite(P):
                    return P

                # Increase regularization factor for the next iteration
                eps *= increment_factor

            raise ValueError("Unable to convert the matrix to positive definite within max iterations.")

        def sqrt_func(P):
            '''
            Define a sqrt funct to solve numpy.linalg.LinAlgError: 6-th leading minor of the array is not positive definite
            The sulutions is suggested by https://github.com/rlabbe/filterpy/issues/62
            If it fails, adjust the matrix and try again by get_near_psd.
            '''
            
            if np.any(np.isinf(P)) or np.any(np.isnan(P)):
                raise ValueError("The matrix contains inf, none or nan.")

            try:
                result = scipy.linalg.cholesky(P)
            except scipy.linalg.LinAlgError:
                P = get_near_psd(P)
                result = scipy.linalg.cholesky(P)

            return result

        points = MerweScaledSigmaPoints(n=12, 
                                        alpha=1.5, 
                                        beta=0, 
                                        kappa=2,
                                        sqrt_method=sqrt_func,
                                        subtract=residual_x)  # Adjust these parameters as needed
        
        self.kf = UnscentedKalmanFilter(dim_x=12, 
                                        dim_z=4, 
                                        dt=1, fx=self.kf_fx, 
                                        hx=self.kf_hx_gps_front, 
                                        points=points,
                                        x_mean_fn=custom_x_mean_fn,
                                        z_mean_fn=custom_z_mean_fn,
                                        residual_x=residual_x,
                                        residual_z=residual_z)
        
        # Kalman filter initial state
        self.orientation_kf = 0  # [deg] the vectors azimuth agle of physical tail to head. angle of N72.
        self.cov_matrix_h = None

        # Kalman filter initial state
        if self.east_h:
            self.kf.x = np.array([self.east_h,    # 0 east
                                  self.north_h,   # 1 north
                                  self.east_h,    # 2 east of second GPS
                                  self.north_h,   # 3 north of second GPS
                                  self.speed,   # 4 velocity [m/s]
                                  0.01,         # 5 acc [m/s^2]
                                  0,            # 6 heading front gps [rad]
                                  0,            # 7 rate of turning front GPS[rad/s]
                                  0,            # 8 heading back gps [rad]
                                  0,            # 9 rate of turning back gps [rad/s]
                                  0,            # 10 orientation [deg]
                                  0,            # 11 offset [s]
                                  ])
        else:
            self.kf.x = np.array([1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.])

        # The uncertainty of the initial state
        self.kf.P = np.diag([2500, 2500, 2500, 2500, 250, 4, 3, 0.1, 3, 0.1, 10000, 25])


    def kf_fx(self, x, dt):
        '''
        State transition matrix/function for the Kalman filter.

        Core to kf.x

        This is a nonlinear process, so we use the Unscented Kalman Filter (UKF) to handle it.
        '''

        # Assuming a simple constant velocity model for demonstration
        # Extract state components for readability
        x_east, y_north, x_east_2, y_north_2, velocity, acc, heading_1, rate_of_turn_1, heading_2, rate_of_turn_2, orientation, lag = x

        # (1) Update velocity based on acceleration, Clip the acc/velocity to a reasonable range
        acc = np.clip(acc, -1.5, 0.8)  # max acceleration 1 m/s^2
        
        # add stationary mode
        if velocity < 0.01:
            velocity = 0

        rate_of_turn_1 = np.clip(rate_of_turn_1, -0.025, 0.025)  # max rate of turn 10 rad.s^-1
        rate_of_turn_2 = np.clip(rate_of_turn_2, -0.025, 0.025)  # max rate of turn 10 rad.s^-2

        # (2) Update headings based on rate of turn
        new_heading_1 = (heading_1 + rate_of_turn_1 * dt + np.pi) % (2 * np.pi) - np.pi
        new_heading_2 = (heading_2 + rate_of_turn_2 * dt + np.pi) % (2 * np.pi) - np.pi

        # (3) Update positions based on velocity and heading
        # For the front GPS (GPS 1)
        de_1 = (velocity + 0.5 * dt * acc) * np.sin(heading_1 + 0.5 * rate_of_turn_1 * dt) * dt
        dn_1 = (velocity + 0.5 * dt * acc) * np.cos(heading_1 + 0.5 * rate_of_turn_1 * dt) * dt

        # For the rear GPS (GPS 2), assuming it follows the same velocity but might have a different heading
        de_2 = (velocity + 0.5 * dt * acc) * np.sin(heading_2 + 0.5 * rate_of_turn_2 * dt) * dt
        dn_2 = (velocity + 0.5 * dt * acc) * np.cos(heading_2 + 0.5 * rate_of_turn_2 * dt) * dt

        new_velocity = np.clip(velocity + acc * dt, 0, 100)  # max speed 100m/s

        
        # (4) Update orientation/azimuth angle [degree] of the vector of physical tail to head
        #new_orientation = (np.degrees(np.arctan2((y_north_2 - y_north - dn_1 + dn_2), (x_east_2 - x_east - de_1 + de_2))) + 360) % 360

        new_orientation = (np.degrees(np.arctan2((x_east - x_east_2 + de_1 - de_2), (y_north - y_north_2 + dn_1 - dn_2))) + 360) % 360
        
        # (5) Update lag. The lag does not work for low speed.
        #new_lag = self.length / (velocity + 0.5 * acc * dt) 
        new_offset = np.sqrt((x_east_2 - x_east)**2 + (y_north_2 - y_north)**2)

        new_offset = np.clip(new_offset, 100-30, 100+30)

        # Update the state vector with new values
        x_updated = x.copy()
        x_updated[0] += de_1  # Update east position of GPS 1
        x_updated[1] += dn_1  # Update north position of GPS 1
        x_updated[2] += de_2  # Update east position of GPS 2
        x_updated[3] += dn_2  # Update north position of GPS 2
        x_updated[4] = new_velocity  # Update velocity
        x_updated[5] = acc  # Update acceleration
        x_updated[6] = new_heading_1  # Update heading for GPS 1
        x_updated[7] = rate_of_turn_1  # Update rate of turn for GPS 1
        x_updated[8] = new_heading_2  # Update heading for GPS 2
        x_updated[9] = rate_of_turn_2  # Update rate of turn for GPS 2
        x_updated[10] = new_orientation  # Update orientation
        x_updated[11] = new_offset  # Update lag

        # exclude the negative value
        for i in range(5):
            if x_updated[i] < 0:
                # x[0] ==1 will tigger the reset by measurements.
                x_updated[i] = 1
                
        return x_updated
    
    def kf_hx_gps_front(self, x):
        '''
        Measurement matrix/function for the Kalman filter. Turning the state space into the measurement space.

        Core to kf.x
        '''
        # Unpack the state vector
        x_east, y_north, x_east_2, y_north_2, velocity, acc, heading_1, rate_of_turn_1, heading_2, rate_of_turn_2, orientation, offset = x

        return [x_east, y_north, velocity, heading_1]
    
    def kf_hx_gps_back(self, x):

        # Unpack the state vector
        x_east, y_north, x_east_2, y_north_2, velocity, acc, heading_1, rate_of_turn_1, heading_2, rate_of_turn_2, orientation, offset = x

        return [x_east_2, y_north_2, velocity, heading_2]
    
    def kf_hx_beacon(self, x):
        return [x[0], x[1]]
    
    def kf_Q(self,dt):
        '''
        Process noise matrix for the Kalman filter.

        TODO: Design Q matrix based on the train's dynamics. Check Page 357

        '''
        
        return Q_discrete_white_noise_two_gps(dt=dt,
                                              var_acc=0.05,
                                              var_theta_1=0.1,
                                              var_theta_2=0.1)
        
    def init_orientation(self, df):
        '''
        Orientation of the train.

        When speed is low, the heading is hightly uncertain. 
        Do not compare it with orientation.
        '''
        # Determine the primary direction of movement
        # Compute the differences without taking absolute values
        df['time_difference'] = df['timestamp'].diff().dt.total_seconds().fillna(0)

        # Calculate the next 'e' and 'n' values based on the current speed and heading
        df['next_e'] = df['east'] + df['time_difference'] * df['speed'] * np.sin(np.radians(df['heading']))
        df['next_n'] = df['north'] + df['time_difference'] * df['speed'] * np.cos(np.radians(df['heading']))

        # Calculate the difference in 'n' and 'e' between the current and previous row
        # if systemid_d == 0, the differnce is likely zero, becasue the movement 'next' has been removed.
        # if systemid_d == 5, the difference is likely casued by offset.
        # if systemid_d == -5, the difference is likely casued by offset.
        df['difference_n'] = df['north'] - df['next_n'].shift(1)
        df['difference_e'] = df['east'] - df['next_e'].shift(1)
        df['offset'] = np.sqrt(df['difference_n']**2 + df['difference_e']**2)

        # Calculate the orientation using the arctan2 function and normalize it to the range 0 to 360
        # if systemid_d == 0, the orientation is nonsense.
        # if systemid_d == 5, the orientation is from 7 to 2. N27 = arctan(E7-E2/N7-N2)
        # if systemid_d == -5, the orientation is from 2 to 7. N72 = arctan(E2-E7/N2-N7)
        df['orientation'] = (np.degrees(np.arctan2(df['difference_e'], df['difference_n'])) + 360) % 360
        df['orientation_heading'] = (df['orientation'] - df['heading'] + 360) % 360

        # Calculate the difference in 'systemid' between the current and previous row, it could be 0, -5, 5 for icemera.
        df['systemid_d'] = df['systemid'] - df['systemid'].shift(1)

        offset = 0.5 * (df[df['systemid_d'] == -5].offset.mean() + df[df['systemid_d'] == 5].offset.mean())
        # The angle of vector 7-2 (tail-head) relative to N
        orientation_N72 = circmean(df[df['systemid_d'] == -5]['orientation'], low=0, high=360)
        # The angle of vector 2-7 (head-tail) relative to N
        orientation_N27 = circmean(df[df['systemid_d'] == 5]['orientation'], low=0, high=360)

        # update
        self.orientation_N72 = orientation_N72 if orientation_N72 is not None else self.orientation_N72
        self.orientation_N27 = orientation_N27 if orientation_N27 is not None else self.orientation_N27
        self.length = offset if offset is not None else self.length
            
        count_not_invert = df[
            (df['systemid_d'] == -5) &
            (df['speed'].between(10, 100)) &
            (df['orientation_heading'].between(-45, 45))
        ].shape[0]

        count_invert = df[
            (df['systemid_d'] == -5) &
            (df['speed'].between(10, 100)) &
            (
                (df['orientation_heading'].between(-180, -135)) |
                (df['orientation_heading'].between(135, 180))
            )
        ].shape[0]

        self.inverted += count_invert - count_not_invert

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
        self.um_train_number = []   # int
        self.um_position = []   # str
        self.total_distance = []  # int
        self.inverted = 0

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
            self.inverted = 0  # Default value
            for i in circulation_record['Positioning']:
                if i == 'Inverted':
                    self.inverted += 1  # Assign 1 if 'Inverted' is found
                elif i == 'Not Inverted':
                    self.inverted -= 1
                else:
                    pass
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

        # (1) Q
        # TODO: Adjust the values according to HDOP, VDOP, PDOP, and TDOP
        R_gps_front = np.diag([5**2, 5**2, (0.5)**2, (0.01)**2])  # Assuming 5m position error, 0.5 m/s speed error, 0.6 degree heading error
        R_gps_back = np.diag([5**2, 5**2, (0.5)**2, (0.01)**2])  # Assuming 5m position error, 0.5 m/s speed error, 0.6 degree heading error
        
        # TODO: Adjust the values according to the beacon's accuracy
        R_beacon = np.diag([10**2, 10**2])  # Assuming 10m position error for beacon

        # TODO: (1) how to deal with a list of boolean values. (2) How to let KF realized it is inverted and add it in condtion.
        
        # (2) Orientation
        # deal with orientation. There are chance that the initial orientation fails when GNSS missing, inadequate.
        if len(new_gnss) > 20:
            self.init_orientation(new_gnss)

        if self.inverted > 0:
            back = self.kf_hx_gps_front
            front = self.kf_hx_gps_back
            R_back = R_gps_front
            R_front = R_gps_back
        else:
            front = self.kf_hx_gps_front
            back = self.kf_hx_gps_back
            R_front = R_gps_front
            R_back = R_gps_back

        # (3) Process GNSS data
        if new_gnss is not None and not new_gnss.empty:
            # Process GNSS data into DataFrame
            new_gnss.loc[:, 'sensor_type'] = 'GNSS'
            new_gnss.loc[:, 'sensor_id'] = new_gnss['systemid']
            new_gnss.loc[:,'H'] = [front if i%2 == 0 else back for i in new_gnss['sensor_id']]
            new_gnss.loc[:,'R'] = [R_front if i%2 == 0 else R_back for i in new_gnss['sensor_id']] 
            new_gnss.loc[:,'z'] = [(row['east'], row['north'], row['speed'], np.deg2rad(row['heading'])) for index, row in new_gnss.iterrows()]
            combined_data.append(new_gnss[['timestamp', 'sensor_type', 'sensor_id', 'H', 'R', 'z']])

        # (4) Process Beacon data
        if new_beacons is not None and not new_beacons.empty:
            new_beacons.loc[:,'sensor_type'] = 'Beacon'
            new_beacons['sensor_id'] = new_beacons.index.get_level_values('primary_location_code')
            new_beacons.loc[:,'H'] = [self.kf_hx_beacon] * len(new_beacons)
            new_beacons.loc[:,'R'] = [R_beacon] * len(new_beacons)
            new_beacons.loc[:,'z'] = [(row['east'], row['north']) for index, row in new_beacons.iterrows()]
            combined_data.append(new_beacons[['timestamp','sensor_type', 'sensor_id', 'H', 'R', 'z']])

        # (5) Merge and sort combined data
        try:    
            if combined_data != []:
                
                combined_df = pd.concat(combined_data).sort_values(by='timestamp')

                # Calculate dt and filter out delayed updates
                # Drop signlas comes in delayed
                process_df = combined_df[combined_df['timestamp'] > self.previous_timestamp].copy()
                
                # Filter duplicated signals
                process_df = process_df.drop_duplicates(subset=['sensor_type', 'sensor_id', 'z'], keep='first')

                # Set the initial dt to a reasonable value if it's too large (first time init) or negative
                init_dt = (process_df['timestamp'].iloc[0] - self.previous_timestamp).total_seconds()  # Initial dt of this batch
                init_dt = init_dt if ((init_dt > 0) and (init_dt < 1000)) else 0

                process_df['dt'] = process_df['timestamp'].diff().dt.total_seconds().fillna(init_dt)
                
                # The initialization need none zero dt because dt are sometimes used as a denominator in Q.
                process_df['dt'] = process_df['dt'].replace(0, 0.001)

                # Construct Fs based on dt
                zs = process_df['z'].tolist()
                Hs = process_df['H'].tolist()
                Rs = process_df['R'].tolist()
                Fs = [self.kf_fx] * len(process_df)
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
                            content=f"Signal comes in delay, and been drop from the batch process:\n"
                            f"{unprocessed['timestamp', 'sensor_type', 'sensor_id']}",
                            status="Skiped",
                            value=len(unprocessed),
                            level="INFO"
                        )

                return zs, Hs, Rs, Fs, Qs, metadata
            
            else:
                return None
        except Exception as e:
            print(f"Error in transform_to_matrix: {e}")


    def transform_to_df(self, mu, cov, metadata: pd.DataFrame, zs):
        '''
        Transform the Kalman filter output into two DataFrames:
        1. A DataFrame with latitude, longitude, heading, and speed along with their covariances and metadata.
        2. An extended DataFrame that includes all states and covariances from the Kalman filter output along with the metadata.
        '''

        # Extract the means for east, north, v_east, and v_north
        east_1 = mu[:, 0].flatten()  # Estimated east position of GPS 1
        north_1 = mu[:, 1].flatten()  # Estimated north position of GPS 1
        east_2 = mu[:, 2].flatten()  # Estimated east position of GPS 2
        north_2 = mu[:, 3].flatten()  # Estimated north position of GPS 2
        velocity = mu[:, 4].flatten()  # Estimated velocity [m/s]
        acceleration = mu[:, 5].flatten()  # Estimated acceleration [m/s^2]
        heading_1 = mu[:, 6].flatten()  # Estimated heading of GPS 1 [rad]
        rate_of_turn_1 = mu[:, 7].flatten()  # Estimated rate of turning of GPS 1 [rad/s]
        heading_2 = mu[:, 8].flatten()  # Estimated heading of GPS 2 [rad]
        rate_of_turn_2 = mu[:, 9].flatten()  # Estimated rate of turning of GPS 2 [rad/s]
        orientation = mu[:, 10].flatten()  # Estimated orientation [deg]
        lag = mu[:, 11].flatten()  # Should be Estimated lag [s], now offset

        try:
            longitude_h, latitude_h = self.to_lonlat.transform(east_1, north_1)
            longitude_t, latitude_t = self.to_lonlat.transform(east_2, north_2)

        except Exception as e:
            print(f"Error converting UTM to lat/lon: {e} \n{east_1} \n{north_1}")
            
        # Create a simplified DataFrame for results
        results = pd.DataFrame({
            'latitude_h': latitude_h,
            'longitude_h': longitude_h,
            'latitude_t': latitude_t,
            'longitude_t': longitude_t,
            'east_h': east_1,
            'north_h': north_1,
            'east_t': east_2,
            'north_t': north_2,
            'speed': velocity,
            'heading_h': (np.degrees(heading_1)+360)%360,  # Convert to degrees and normalize to [0, 360),
            'heading_t': (np.degrees(heading_2)+360)%360,  # Convert to degrees and normalize to [0, 360),
            'acceleration': acceleration,
            'rate_of_turn': (np.degrees(rate_of_turn_1)+360)%360,
            'orientation': orientation,
            'lag': lag
        })

        # add raw
        results['raw_east'] = [z[0] for z in zs]
        results['raw_north'] = [z[1] for z in zs]
        results['raw_speed'] = [z[2] if len(z) >= 4 else None for z in zs]
        results['raw_heading'] = [(np.degrees(z[3])+360)%360 if len(z) >= 4 else None for z in zs]

        # Add covariance for latitude, longitude, speed, heading
        results['cov_east_h'] = cov[:, 0, 0]  # Proxy for E covariance
        results['cov_north_h'] = cov[:, 1, 1]  # Proxy for N covariance
        results['cov_east_t'] = cov[:, 2, 2]
        results['cov_north_t'] = cov[:, 3, 3]
        results['cov_speed'] = cov[:, 4, 4]  # Proxy for speed covariance
        results['cov_heading_h'] = (np.degrees(cov[:, 6, 6])+360)%360
        results['cov_heading_t'] = (np.degrees(cov[:, 8, 8])+360)%360  # Proxy for heading covariance

        # Merge simplified results with metadata
        metadata_subset = metadata[['timestamp', 'sensor_type', 'sensor_id', 'dt']]
        results = pd.concat([metadata_subset.reset_index(drop=True), results.reset_index(drop=True)], axis=1)

        return results, metadata, mu, cov

    
    def update(self, new_gnss: Union[pd.DataFrame, None], new_beacons: Union[pd.DataFrame, None]):
        '''
        Method to update the train object with data from GPS or Beacons.
        '''
        # turn new_gnss and into new_beacons zs matrix, Hs matrix, Rs matrix in ASC order by timestamp
        raw = self.transform_to_matrix(new_gnss, new_beacons)
        
        if raw is not None:
            zs, Hs, Rs, Fs, Qs, df_meta = raw
            # Kalman filter return a matrix of states, and covariance matrix
            mu, cov = self.kf.batch_filter(zs, Fs=Fs, Rs=Rs, dts= df_meta['dt'], Hs=Hs, Qs=Qs)
            results, meta,_ ,_ = self.transform_to_df(mu, cov, df_meta, zs)

            # Only update the train with the last position from the Kalman filter
            # .item() is to convert the numpy types to native Python types that are JSON serializable.
            if not results.empty:
                self.latitute_h = results['latitude_h'].iloc[-1].item()
                self.longitude_h = results['longitude_h'].iloc[-1].item()
                self.east_h = results['east_h'].iloc[-1].item()
                self.north_h = results['north_h'].iloc[-1].item()
                self.latitute_t = results['latitude_t'].iloc[-1].item()
                self.longitude_t = results['longitude_t'].iloc[-1].item()
                self.east_t = results['east_t'].iloc[-1].item()
                self.north_t = results['north_t'].iloc[-1].item()
                self.speed = results['speed'].iloc[-1].item()
                self.heading_h = results['heading_h'].iloc[-1].item()
                self.heading_t = results['heading_t'].iloc[-1].item()
                self.orientation_kf = results['orientation'].iloc[-1].item()
                self.cov_matrix_h = [results['cov_east_h'].iloc[-1].item(), 
                                     results['cov_north_h'].iloc[-1].item(), 
                                     results['cov_speed'].iloc[-1].item(), 
                                     results['cov_heading_h'].iloc[-1].item()]
                self.cov_matrix_t = [results['cov_east_t'].iloc[-1].item(),
                                     results['cov_north_t'].iloc[-1].item(),
                                     results['cov_speed'].iloc[-1].item(),
                                     results['cov_heading_t'].iloc[-1].item()]
                # reset the inverted value when the speed is low.
                if self.speed < 1:
                    self.inverted = self.inverted/2

                # Contribute two inverted value, exclude the low speed when the heading is not accurate.
                if self.orientation_kf and self.speed > 10:
                    orientation_diff = abs(self.orientation_kf - self.heading_h)
                    if orientation_diff <= 45 or orientation_diff >= 315:
                        self.inverted -= 1
                    elif orientation_diff >= 135 and orientation_diff <= 225:
                        self.inverted += 1
        else:
            # No new data to update the train with
            return None, None

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

        return results, [meta, mu, cov, zs]


    def to_dict(self):
        '''
        Export to json serializable dictionary for FastAPI response.
        '''


        # Define a dictionary to hold serializable attributes
        serializable_dict = {}
        
        # List of attributes to be included in the dict
        attrs_to_include = ['rame_id', 'rame_number', 'systemid', 'train_number', 'traveldate', 
                            'origin', 'destination', 'status', 'type','inverted' ,'um_train_number', 
                            'um_position', 'last_gps_t', 'last_beacon_t', 'last_beacon_code', 
                            'lock', 'satellites', 'cov_matrix_h', 'latitute_h', 'longitude_h', 'orientation_kf',
                            #'altitude', 
                            'speed', 'heading_h','heading_t']
        # Number of decimal places to round floating-point numbers to
        decimal_places = 6

        # Iterate over specified attributes
        for attr in attrs_to_include:
            attr_value = getattr(self, attr, None)

            # Check if it's a numpy type and handle floats separately for rounding
            if isinstance(attr_value, np.generic):
                if np.issubdtype(attr_value, np.floating):
                    # Check for finite values before converting and rounding
                    if np.isfinite(attr_value):
                        serializable_dict[attr] = round(attr_value.item(), decimal_places)
                    else:
                        serializable_dict[attr] = None  # or some default value for non-finite
                else:
                    serializable_dict[attr] = attr_value.item()
            # Handle regular float values, rounding and checking for finiteness
            elif isinstance(attr_value, float):
                if np.isfinite(attr_value):
                    serializable_dict[attr] = round(attr_value, decimal_places)
                else:
                    serializable_dict[attr] = None  # or some default value for non-finite
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
            'latitute': self.latitute_h,
            'longitude': self.longitude_h,
            'altitude': self.altitude,
            'speed': self.speed,
            'heading': self.heading_h,
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

def initialize_or_update_train(sys, item, length=195):
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
        train.to_en = sys.to_en
        train.to_lonlat = sys.to_lonlat
        train.length = length
        train.previous_timestamp = pd.Timestamp('1900-01-01', tz='UTC')
    else:
        # If already in table , update the existing Train object
        train = sys.trains_mapping_dict[train_id]
        train.rame_id = train_id
        train.rame_number = rame_number
        train.systemid = systemids
        train.to_en = sys.to_en
        train.to_lonlat = sys.to_lonlat
        train.previous_timestamp = pd.Timestamp('1900-01-01', tz='UTC')

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
        df['east'], df['north'] = sys.to_en.transform(df['longitude'].values,df['latitude'].values)
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
              f" Now the query datetime: {sys.last_gps_t}.",
              f" The length of records: {len(gps_updates) + len(beacon_updates)}.")
        
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
        gps_df = create_gps_updates_df(gps_updates, sys)

    if beacon_updates:
        sys.last_beacon_t = beacon_updates[-1]['LocationDateTime']
        sys.length_beacon = len(beacon_updates)
        beacon_df = create_beacon_updates_df(beacon_updates, sys.beacon_mapping_df)

    # Update qurey window
    sys.end_timestamp = max(sys.last_gps_t, sys.last_beacon_t) + sys.query_step
        
    # (3) send update to trains
    try:
        for train in sys.trains_dict.values():
            gps_data = gps_df[gps_df['systemid'].isin(train.systemid)].copy() if not gps_df.empty else None
            beacon_data = beacon_df[beacon_df['train_number'].isin(train.train_number)].copy() if not beacon_df.empty else None

            train.update(gps_data, beacon_data)

    except KeyError as e:
        # Log the error with more details
        print(f"KeyError occurred: {e}")
        print("DataFrame before error:")
        print(beacon_data)  # Adjust this line as needed to log more or less data
        raise e

def create_beacon_updates_df(beacon_updates, beacon_mapping_df = None):
    '''
    Create a DataFrame for beacon updates.
    '''
    try:
        df = pd.DataFrame(beacon_updates).copy()
        df = df.rename(columns={
            'OTI_OperationalTrainNumber': 'OTI_train_number',
            'ROTN_OTI_OperationalTrainNumber': 'ROTN_OTI_train_number',
            'LocationPrimaryCode': 'primary_location_code',
            'LocationDateTime': 'timestamp',
            'CountryCodeISO': 'country_code',
            'MessageDateTime': 'message_datetime',
            'AgainstBooked': 'against_booked'
        })

        # Drop Nan values
        columns_to_check = ['country_code','primary_location_code','timestamp']
        df = df.dropna(subset=columns_to_check).copy()

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

def create_gps_updates_df(gps_updates, sys):
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

    # Ensure latitude and longitude are of type float
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)
    df['speed'] = df['speed'].astype(float)
    df['heading'] = df['heading'].astype(float)

    # TODO: add a gate
    columns_to_check = ['latitude','longitude','heading', 'timestamp']
    df = df.dropna(subset=columns_to_check).copy()

    speed_threshold = (0, 400)  # over 400 m/s is not possible
    df = df[(df['speed'] >= speed_threshold[0]) & (df['speed'] <= speed_threshold[1])].copy()

    # Add UTM coordinates to the DataFrame
    df['east'], df['north'] = sys.to_en.transform(df['longitude'].values, df['latitude'].values)

    return df


async def historical_train_positions(rame_id: int,
                                     sys) -> List:
    '''
    Fetch historical train positions for a given date and train id, and send it to KF for localization.
    '''
    data = []  # Set data to an empty list if it is None

    # to be processed by Kalman filter
    gps_raw, beacon_raw = await asyncio.gather(
            fetch_gps(sys.trains_dict[rame_id].systemid, sys.system_date, sys.end_timestamp),
            fetch_beacon(sys.trains_dict[rame_id].train_number, sys.system_date, sys.end_timestamp)
            )
    if len(gps_raw) == 0 and len(beacon_raw) == 0:
        return []
    
    else:
        gps_df = create_gps_updates_df(gps_raw, sys) if gps_raw else None
        beacon_df = create_beacon_updates_df(beacon_raw, sys.beacon_mapping_df) if beacon_raw else None
        
        # Return the resutls of the KF
        results, [meta, mu, cov, zs] = sys.trains_dict[rame_id].update(gps_df, beacon_df)
        results['raw_longitude'], results['raw_latitude'] = sys.trains_dict[rame_id].to_lonlat.transform(results['raw_east'], results['raw_north'])
        
        return results.to_json(orient='records', 
                               date_format='iso',
                               double_precision=6, 
                               force_ascii=False)

async def get_realtime_train_data(trains_dict: dict, rame_id: int = None) -> List: # type: ignore
    # Return the latest attributes of train as a list of dictionary
    return [train.to_dict() for train in trains_dict.values() if rame_id is None or train.rame_id == rame_id]


async def get_realtime_train_data_geolocation(trains_dict: dict, rame_id: int = None) -> List: # type: ignore
    # Return the train data as a dictionary
    return [train.to_dict_geo() for train in trains_dict.values() if rame_id is None or train.rame_id == rame_id]
