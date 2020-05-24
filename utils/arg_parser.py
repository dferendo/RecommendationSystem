import json
import argparse


def extract_args_from_json():
    parser = argparse.ArgumentParser(description='Recommendation Systems')

    parser.add_argument('--json_configs', nargs="?", type=str, default=None, help='')
    args = parser.parse_args()

    with open(args.json_configs) as f:
        arguments_dict = json.load(fp=f)

    return arguments_dict
