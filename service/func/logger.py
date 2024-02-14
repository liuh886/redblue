import logging
import os
from logging.handlers import TimedRotatingFileHandler
import json
from datetime import datetime, timedelta
from typing import Dict, Any

class StructuredLogger:
    def __init__(self, app_name, log_dir="./logs"):
        self.app_name = app_name
        self.log_dir = log_dir
        self.ensure_log_dir_exists()
        self.logger = self.setup_logger()

    def ensure_log_dir_exists(self):
        os.makedirs(self.log_dir, exist_ok=True)

    def setup_logger(self):
        logger = logging.getLogger(self.app_name)
        logger.setLevel(logging.INFO)

        handler = TimedRotatingFileHandler(
            filename=os.path.join(self.log_dir, f"{self.app_name}.log"),
            when="midnight",
            backupCount=30
        )
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def log(self, 
            event: str, 
            object_name: str, 
            object_type: str,
            content: str, 
            status: str, 
            value: str, 
            level="INFO"):
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "app_name": self.app_name,
            "event": event,
            "object_name": object_name,
            "object_type": object_type,
            "content": content,
            "status": status,
            "value": value,
            "level": level
        }
        log_message = json.dumps(log_entry)
        if level.upper() == "ERROR":
            self.logger.error(log_message)
        elif level.upper() == "WARNING":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    # Additional methods to handle status updates and avoid duplicating active issues can be added here.


class SystemStatus:
    '''
    To record the system state of the service.
    '''

    def __init__(self):
        self.system_date = datetime.now()  # - timedelta(hours=1)
        self.last_gps_t: datetime = None
        self.last_beacon_t: datetime = None
        self.trains_dict: Dict[int, Any] = {}  # Assuming train IDs are integers for the keys
        self.beacon_mapping_table: Dict[int, Any] = {}
        self.updating = True
        self.trains_mapping_table = {}
        self.beacon_country = []
        self.beacon_dict = {}
        self.last_log_query_time = datetime.min
        self.logger = None
        self.length_gps: int = None
        self.length_beacon: int = None

    def update_beacon_mapping_table(self, beacon_id: int, beacon_data: Any):
        self.beacon_mapping_table[beacon_id] = beacon_data

    def reset_trains_dict(self):
        self.trains_dict.clear()

    def log_event_initiation(self, circulation_data: Any):
        '''
        Automatically examination after the initiation of the system.
        '''
        # Check the trains_dict
        if not self.trains_dict:
            self.logger.log(
                event='service_init',
                object_name="trains_dict",
                object_type="dictionary",
                content=f"No train valid in circulation table {self.system_date}, len of circulation: {len(circulation_data)}",
                status="error",
                value="404",
                level="WARNING"
            )
        else:
            self.logger.log(
                event="service_init",
                object_name="trains_dict",
                object_type="dictionary",
                content=f"'initialization complete with train id: {list(self.trains_dict.keys())}",
                status="ok",
                value="200",
                level="INFO"
            )

        # Add other methods to modify or access the state as needed
