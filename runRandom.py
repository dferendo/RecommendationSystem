from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset
from utils.reset_seed import set_seeds


def experiments_run():
    configs = extract_args_from_json()
    set_seeds(configs['seed'])

    df_train, df_val, df_test = split_dataset(configs)

    # random_model = baselines.RandomSlateGeneration(5, np.arange(500))


if __name__ == '__main__':
    experiments_run()
