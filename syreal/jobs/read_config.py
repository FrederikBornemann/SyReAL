import yaml
import os
import string
from pathlib import Path

PKG_DIR = Path(__file__).parents[2]
CONFIG = PKG_DIR / "config.yml"

def extract_strings_recursively(d):
    if not isinstance(d, dict):
        return {d: d}
    result = {}
    for k, v in d.items():
        if not isinstance(v, dict):
            result[k] = v
        elif isinstance(v, dict):
            result.update({k2: v2 for k2, v2 in extract_strings_recursively(v).items()})
    return result

def substitute_vars(data, vars):
    if isinstance(data, str):
        #print("str: ", data)
        template = string.Template(data)
        return template.safe_substitute(extract_strings_recursively(vars))
    elif isinstance(data, dict):
        for key in data:
            #print("key: ", key, type(key))
            data[key] = substitute_vars(data[key], vars)
        return data
    elif isinstance(data, list):
        for i, item in enumerate(data):
            data[i] = substitute_vars(item, vars)
        return data
    else:
        return data

def read_config(path=CONFIG):
    try:
        with open(path, 'r') as yml:
            config = yaml.safe_load(yml)
    except Exception:
        raise Exception("File not found")
        return
    else:
        return substitute_vars(config, config)
    return None
