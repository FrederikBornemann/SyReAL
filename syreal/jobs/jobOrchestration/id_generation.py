import hashlib


# this function generates a unique ID for a number of parameters and returns the first n characters of the hash
def generate_id(*params, n=8):
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
