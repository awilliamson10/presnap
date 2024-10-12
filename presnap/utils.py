import json


def load_vocab(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)