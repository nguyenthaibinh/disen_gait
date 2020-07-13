from __future__ import print_function
from pathlib import Path
import yaml
from time import time
from datetime import datetime as dt

def get_current_time(format="%Y-%m-%d-%H-%M-%S"):
    cur_time = dt.utcfromtimestamp(time()).strftime(format)
    return cur_time

class ConfigLoader(object):
    def __init__(self, config_file):
        # Load common configs
        ROOT_DIR = Path(__file__).resolve().parents[2]
        CONFIG_FILE = Path(ROOT_DIR, config_file)
        with open(CONFIG_FILE, 'r') as stream:
            self.config = yaml.safe_load(stream)