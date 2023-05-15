# This file contains the functions used in the job orchestration module.
import timeit
import hashlib
import json


def generate_id(lst, n=8) -> str:
    """Generate a unique id from a list of objects."""
    # Convert the list to a JSON string
    json_str = json.dumps(lst, sort_keys=True)
    # Generate a hash object from the JSON string
    hasher = hashlib.sha256()
    hasher.update(json_str.encode('utf-8'))
    # Return the first n characters of the hash digest
    return hasher.hexdigest()[:n]


def Logger(add_handler=True):
    """Create a logger that logs to a file and returns the logger and the file descriptor"""
    import logging
    from constants import PID_LOG_FILE
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(PID_LOG_FILE)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if add_handler:
        logger.addHandler(handler)
    keep_fds = [handler.stream.fileno()]
    return logger, keep_fds


def timeit_wrapper(func):
    """Decorator to time a function."""
    logger, keep_fds = Logger(add_handler=False)

    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        logger.info("Function {} took {} seconds to execute.".format(
            func.__name__, end_time - start_time))
        print("Function {} took {} seconds to execute.".format(
            func.__name__, end_time - start_time))
        return result
    return wrapper
