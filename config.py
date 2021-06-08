import json


class Config(dict):
    def __init__(self, config_file_path):
        super(Config, self).__init__()
        with open(config_file_path, "r") as f:
            config_params = json.load(f)  # this is a dictionary
        self.__dict__ = config_params
