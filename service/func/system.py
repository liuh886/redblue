import logging
import os
from logging.handlers import TimedRotatingFileHandler
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import time
from pandas import DataFrame
import pytz

class SystemStatus:
    '''
    The environment for the system.
    '''

    def __init__(self, 
                 system_date = datetime.now(),
                 query_step = timedelta(minutes=30),
                 utm_zone:int = 31,
                 updating:bool = True,
                 local_timezone = pytz.timezone('Europe/Paris'),
                 system_timezone = pytz.timezone('UTC'),
                 beacon_country:list = ["GB", "FR", "BE", "NL", "DE"]):

        # Your local machine time zone
        self.local_timezone = local_timezone
        # The time zone of the service
        self.system_timezone = system_timezone
        # Use the provided local time to set the initial values
        self.system_date = self.validate_system_date(system_date)

        # set the UTM zone
        self.utm_zone = utm_zone
        self.updating = updating
        self.beacon_country = beacon_country
        
        # The query time when the system is initiated, than will be updated
        self.last_gps_t = self.system_date
        self.last_beacon_t = self.system_date
        self.query_step = query_step
        self.end_timestamp = max(self.last_gps_t, self.last_beacon_t ) + self.query_step

        # Key tables
        self.trains_dict: Dict[int, Any] = {}  # RameId: are integers for the keys
        self.trains_mapping_dict: Dict[int, Any] = {} 
        self.beacon_country = beacon_country
        self.beacon_dict = {}
        self.beacon_mapping_df: Optional[DataFrame] = None

        # Sub-component
        self.logger = None
        self.last_log_query_time = datetime.min

        # Status of the last cycle
        self.length_gps: Optional[int] = None
        self.length_beacon: Optional[int] = None


    def validate_system_date(self, system_date) -> datetime:
        '''
        Ensure the system date is a datetime object with a timezone in UTC.
        '''

        ## Ensure datetime
        if not isinstance(system_date, datetime):
            system_date = datetime.strptime(system_date, '%Y-%m-%d %H:%M:%S') # type: ignore
        
        #  timezone aware by local timezone
        if system_date.tzinfo is None:
            system_date = self.local_timezone.localize(system_date)

        # Ensure convert to system timezone
        if system_date.tzinfo != self.system_timezone:
            system_date = system_date.astimezone(self.system_timezone)

        return system_date
    
    def reset_trains_dict(self):
        self.trains_dict.clear()

    def async_timed(self, func_name):
        '''
        Decorator to measure the time taken by an async function.
        '''

        async def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = await func(*args, **kwargs)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

                if self.logger:
                    self.logger.log(
                        event="function_profile",
                        object_name=func_name,
                        object_type="function",
                        content=f"executed in {elapsed_time:.2f} seconds.",
                        status="ok",
                        value="200",
                        level="INFO"
                    )
                else:
                    print(f"Profile: {func_name} executed in {elapsed_time:.2f} seconds.")
                return result
            return wrapper
        return decorator
    
    def log_event_initiation(self, circulation_data: Any):
        '''
        Automatically examination after the initiation of the system.
        '''

        if self.logger:
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


class SystemLogger:
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

