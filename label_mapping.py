import json

def load_label_mapping(file_path='cat_to_name.json'):
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
