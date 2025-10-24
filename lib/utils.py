from pathlib import Path
import yaml
import json

def makedir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(Path(path).resolve())

def load_config(path):
    with open(path, 'r') as fh:
        return yaml.safe_load(fh)

def save_yaml(obj, path):
    with open(path,'w') as fh:
        yaml.safe_dump(obj, fh)

def load_json(path):
    with open(path, "r") as fh:
        return json.load(fh)

def get_peak_names_from_file(path):
    with open(path, "r") as fh:
        return list(map(lambda x : x.replace('\n', '').replace(":", "_").replace("-", "_"), fh.readlines()))

