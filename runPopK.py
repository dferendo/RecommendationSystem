from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset
from utils.reset_seed import set_seeds
from dataloaders.TestDataLoader import NoAdditionalInfoTestDataLoader
from models.PopK import PopularKSlateGeneration
from utils.experiment_builder_plain import ExperimentBuilderPlain

from torch.utils.data import DataLoader


class ExperimentBuilderPopK(ExperimentBuilderPlain):
    def eval_iteration(self):
        return self.model.forward()


def experiments_run():
    configs = extract_args_from_json()
    print(configs)
    set_seeds(configs['seed'])

    df_train, df_test, df_train_matrix, df_test_matrix, movies_categories = split_dataset(configs)

    test_dataset = NoAdditionalInfoTestDataLoader(df_test, df_test_matrix)
    test_loader = DataLoader(test_dataset, batch_size=configs['test_batch_size'],
                             shuffle=True, num_workers=4, drop_last=True)

    for slate_size in configs['slate_size']:
        print(f'Test for {slate_size}')
        model = PopularKSlateGeneration(slate_size, df_train, df_train_matrix, configs['test_batch_size'])

        experiment_builder = ExperimentBuilderPopK(model, test_loader, len(df_train_matrix.columns), movies_categories,
                                                   configs)
        experiment_builder.run_experiment()


if __name__ == '__main__':
    experiments_run()
