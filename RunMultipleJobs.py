import json
import pandas as pd
import os
import argparse

'''
Example: 
python ./RunMultipleJobs.py --default_configs ./configs/cGAN/default_configs.json --hyper_parameters_tuning ./configs/cGAN/multi_runs.csv --run_file runCGAN.py --run_on_cluster false
'''


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Recommendation Systems')

parser.add_argument('--default_configs', nargs="?", type=str, required=True, help='')
parser.add_argument('--hyper_parameters_tuning', nargs="?", type=str, required=True, help='')
parser.add_argument('--run_file', nargs="?", type=str, required=True, help='')
parser.add_argument('--run_on_cluster', nargs="?", type=str2bool, required=True, help='')
args = parser.parse_args()

bash_script_location = './scripts/run_on_cluster.sh'

with open(args.default_configs, 'r') as json_configs:
    default_config = json.load(json_configs)

with open(args.hyper_parameters_tuning, 'r') as hparams:
    df = pd.read_csv(hparams)

    for _, series in df.iterrows():
        json_string = series.to_json()
        json_string = json.loads(json_string)

        json_merged = {**default_config, **json_string}

        json_merged = json.dumps(json_merged)

        # Sorry for the lazy hack
        json_merged = json_merged.replace(' ', '')
        json_merged = json_merged.replace('\"[', '[')
        json_merged = json_merged.replace(']\"', ']')

        json_merged = json.loads(json_merged)

        # Experiment name (Only mention changeable variables)
        experiment_name = ''
        for key, value in json_string.items():
            if len(key) > 3:
                key = key[:3]

            experiment_name += f'{key}_{value}_'

        json_merged['experiment_name'] = os.path.join(json_merged['experiment_name'], experiment_name)
        json_merged = json.dumps(json_merged)

        if args.run_on_cluster:
            os.system(f"sbatch {bash_script_location} {args.run_file} '{json_merged}'")
        else:
            os.system(f"python {args.run_file} --json_configs_string '{json_merged}'")
