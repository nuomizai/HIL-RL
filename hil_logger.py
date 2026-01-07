from pathlib import Path
import numpy as np
import os

class HILLogger:
    def __init__(self, log_path="./hil_log", name_dict={"is_intervene", "step", "episode", "time", "success"}, log_interval=100):
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path / "hil_log.npy"
        self.name_dict = name_dict
        self.log_interval = log_interval

        # Store time span information
        self.current_time_span = 0
        self.first_time = None
        self.last_time = None

        self.time_offset = None
        self.first_new_time = None
        
        if os.path.exists(self.log_file):
            self.log_data_list = np.load(self.log_file, allow_pickle=True).tolist()

            if self.log_data_list:
                self.resume = True
                self.latest_episode = max(item["episode"] for item in self.log_data_list)
                self.latest_time = max(item["time"] for item in self.log_data_list)

                self.update_time_span()
            else:
                self.resume = False
                self.latest_episode = None
                self.latest_time = None
                
        else:
            self.log_data_list = []

            self.resume = False
            self.latest_episode = None
            self.latest_time = None

        self.log_count = 0
        
    def update_time_span(self):
        """Update current time span information"""
        if self.log_data_list:
            self.first_time = self.log_data_list[0]["time"]
            self.last_time = self.log_data_list[-1]["time"]
            self.current_time_span = self.last_time - self.first_time
        else:
            self.first_time = None
            self.last_time = None
            self.current_time_span = 0

        return self.current_time_span
    
    def log(self, message):
        for key in self.name_dict:
            if key not in message:
                print(f"key {key} not in message")
                exit(0)

        # self.log_data_list.append(message)

        # Create a copy of the message to avoid modifying the original data
        processed_message = message.copy()

        
        if self.resume:
            # Process the episode field
            processed_message["episode"] += self.latest_episode + 1
        
            # Process the time field
            if self.time_offset is None:
                # First time recording new data, set time offset
                self.first_new_time = processed_message["time"]
                self.time_offset = self.first_new_time - self.latest_time
                processed_message["time"] = self.latest_time
            else:
                # Subsequent recording, apply time offset
                processed_message["time"] = processed_message["time"] - self.time_offset
        
        
        # Add to log list
        self.log_data_list.append(processed_message)
        

        if self.log_count % self.log_interval == 0:
            self.save()
        self.log_count += 1
        self.update_time_span()
    
    def close(self):
        self.save()
        self.log_data_list = []
        self.log_count = 0
  
        self.latest_episode = None
        self.latest_time = None
        self.time_offset = None
        self.first_new_time = None

    def save(self):
        np.save(self.log_file, self.log_data_list)
