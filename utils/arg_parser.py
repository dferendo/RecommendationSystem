import json
import argparse


def extract_args_from_json():
    parser = argparse.ArgumentParser(description='Recommendation Systems')

    parser.add_argument('--json_configs', nargs="?", type=str, default=None, help='')
    parser.add_argument('--json_configs_string', nargs="?", type=str, default=None, help='')
    args = parser.parse_args()

    if args.json_configs is not None:
        with open(args.json_configs) as f:
            arguments_dict = json.load(fp=f)
    elif args.json_configs_string is not None:
        arguments_dict = json.loads(args.json_configs_string)
    else:
        raise ValueError('Pass parameters either in file format or string.')

    return arguments_dict
