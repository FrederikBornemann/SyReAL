# this function generates a unique ID for a number of parameters and returns the first n characters of the hash
def generate_id(*params, n=8):
    import hashlib
    # Concatenate the parameters into a single string
    param_string = "".join([str(param) for param in params])
    # Create a SHA-256 hash object
    hash_object = hashlib.sha256()
    # Encode the parameter string as bytes and update the hash object
    hash_object.update(param_string.encode())
    # Get the hexadecimal representation of the hash
    hex_digest = hash_object.hexdigest()
    # Return the first n characters of the hash as the ticket ID
    return hex_digest[:n]

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

import timeit

def timeit_wrapper(func):
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        print("Function {} took {} seconds to execute.".format(func.__name__, end_time - start_time))
        return result
    return wrapper

