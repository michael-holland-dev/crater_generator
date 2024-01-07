import json

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    @classmethod
    def load_config(file_path):
        with json.load(file_path) as file:
            return Config(**file)
    
    def save_config(self, file_path):
        json.dump(self.__dict__(), file_path)
