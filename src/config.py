import json
import os

def get_config():
    cur_dir = os.path.dirname(__file__)

    with open(os.path.join(cur_dir, "config.json"), 'r') as f:
        return json.load(f)
