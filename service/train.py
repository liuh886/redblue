# train.py
class Train:
    def __init__(self, 
                 rame_id, 
                 train_number, 
                 rame_number,
                 traveldate,
                 origin,
                 destination,
                 status,
                 type,
                 um_train_number,
                 um_position):
        
        self.rame_id = rame_id # id
        self.rame_number = rame_number  # trainset number
        self.train_number = train_number # service number
        self.traveldate = traveldate
        self.origin = origin
        self.destination = destination
        self.status = status
        self.type = type
        self.um_train_number = um_train_number
        self.um_position = um_position

        # Add other attributes as needed, e.g., position, speed, etc.

    def update_with_new_data(self, new_gnss, new_beacons):
        # Method to update the train object with new data from ICOMERA or TrainRunningInformation_MainTable_4005
        
        # Sort updates by timestamp, and process position from old to new

        # Update the train's position based on the new data

        ## update matrix from new_gnss and new_beacons


        ## Kalman filter
        self.apply_kalman_filter()

        # Update attributes as needed
        ## GNSS metadata
        self.lock = new_gnss[-1]['metaData.lock']
        self.satellites = new_gnss[-1]['metaData.satellites']
        self.quality = new_gnss[-1]['quality']
        
        ## timestamp metadata
        self.last_gnss_update = new_gnss[-1]['timestamp']
        self.last_beacon_update = new_beacons[-1]['LocationDateTime']
        self.last_beacon_code = new_beacons[-1]['LocationPrimaryCode']
        pass

    def apply_kalman_filter(self):
        # How many input how many output


        # Integrate your Kalman Filter logic here to update the train's estimated position or other states
        
        
        pass
    
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
            'last_gnss_update': self.last_gnss_update,
            'last_beacon_update': self.last_beacon_update,
        }