import yaml
import os
import string
from pathlib import Path

def read_config(path=None, inside=True):
    # when inside src folder and no path specified
    if inside and path == None:
        path = Path(os.getcwd()).parent.absolute()
    # when outside src folder and no path specified
    elif not inside and path == None:
        path = Path(os.getcwd()).absolute()
    try:
        with open(f'{path}/config.yml', 'r') as yml:
            config = yaml.safe_load(yml)
    except Exception:
        raise Exception("File not found")
        return
    else:
        # replace variable placeholders with variable values (${variable} -> variable_value)
        var_dict = dict()
        for section, items in config.items():
            try:
                values = items.values()
            except:
                continue
            for var, value in items.items():
                template = string.Template(value)
                value = template.safe_substitute(var_dict)
                var_dict[var] = value
                config[section][var] = value
        return config
    return None
