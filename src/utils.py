import json

def write_to_json(data, file_path):
    """
    Write data to a JSON file.

    Parameters:
    - data: The data to be written to the JSON file.
    - file_path: The path to the JSON file.
    """
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=2)

def load_from_json(file_path):
    """
    Load data from a JSON file.

    Parameters:
    - file_path: The path to the JSON file.

    Returns:
    - The loaded data.
    """
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data
