import json
import pandas as pd
import os

default_configs = './configs/cGAN/default_configs.json'
hyper_parameters_tuning = './configs/cGAN/multi_runs.csv'
run_file = 'runCGAN.py'

with open(default_configs, 'r') as json_configs:
    default_config = json.load(json_configs)

with open(hyper_parameters_tuning, 'r') as hparams:
    df = pd.read_csv(hparams)

    for _, series in df.iterrows():
        json_string = series.to_json()
        json_string = json.loads(json_string)

        json_merged = {**default_config, **json_string}

        json_merged = json.dumps(json_merged)

        # Sorry for the lazy hack
        json_merged = json_merged.replace('\"[', '[')
        json_merged = json_merged.replace(']\"', ']')

        json_merged = json.loads(json_merged)

        # Experiment name (Only mention changeable variables)
        experiment_name = ''
        for key, value in json_string.items():
            if len(key) > 3:
                key = key[:3]

            experiment_name += f'{key}_{value}_'

        json_merged['experiment_name'] = experiment_name
        json_merged = json.dumps(json_merged)

        os.system(f"python {run_file} --json_configs_string '{json_merged}'")
