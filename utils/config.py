import json

class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        try:
            with open(self.config_file, "r") as file:
                config = json.load(file)
            return config
        except FileNotFoundError:
            return {}

    def save_config(self):
        with open(self.config_file, "w") as file:
            json.dump(self.config, file, indent=4)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()