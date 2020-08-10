from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset
from utils.reset_seed import set_seeds
from dataloaders.TestDataLoader import NoAdditionalInfoTestDataLoader
from models.Random import RandomSlateGeneration
from utils.experiment_builder_plain import ExperimentBuilderPlain

from torch.utils.data import DataLoader
import numpy as np


class ExperimentBuilderRandom(ExperimentBuilderPlain):
    def eval_iteration(self):
        return self.model.forward()


def experiments_run():
    configs = extract_args_from_json()
    print(configs)
    set_seeds(configs['seed'])
    df_train, df_test, df_train_matrix, df_test_matrix, movies_categories, release_years, titles = split_dataset(configs)

    test_dataset = NoAdditionalInfoTestDataLoader(df_test, df_test_matrix)
    test_loader = DataLoader(test_dataset, batch_size=configs['test_batch_size'],
                             shuffle=True, num_workers=4, drop_last=True)

    for movie in [35585,9890,9670,1692,5144,5735,5136,9275,9853,9324,980,5571,1060,3140,817,3970,2505,9508,5147,652,775,4077,474,6122,9925,9909,9870,9905,9574,9796,9907,9791,9787,9804,9786,2581,8929,8150,3517,9882,9836,9875,9938,9439,9720,9459,9615,9927,9780,9900,9928,9885,9684,5501,1348,2244,9910,8146,1009,9409,9934,290,47,7486,3478,4048,4039,4066,5509,5199,3984,3326,4336,2817,3258,6855,7021,4699,8184,3540,4208,7285,3099,4687,3866,7369,4079,5893,8025,7819,6411,7846,5196,4375,6414,3910,2936,5944,7897,7016,6646,6021,7829,486,8626,6867,999,2134,994,8644,541,8418,954,1002,759,1028,2376,7626,8851,1055,8532,2765,6977,1656,1000,269,1027,1043,554,6456,7928,5449,8780,1069,8514,5398,774,7733,7470,4254,1040,1086,1096,8378,8636,4064,4657,3539,5395,6029,5541,4681,3316,4898,3097,5899,4509,5224,7058,4974,7554,6939,7017,7038,4516,4908,4634,3965,5191,6570,2899,5447,5882,4050,3475,4240,4660,3138,6601,4317,3982,6624,6756,6980,7715,5554,6364,2733,3162,3277,8028,8069,6859,6722,4119,6026,5518,7410,8227,5011,7365,8276,6707,7334,5092,7522,7313,7823,5937,6637,6482,6942,5120,7034,4684,3276,7866,5849,6725,7164,6697,5951,5552,3335,5995,6700,4620,4179,8306,7427,6773,7325,5484,7676,6741,7611,4091,7345,6384,6748,4622,4573,3592,7352,8458,6929,4004,5935,4721,8767,5998,6568,5198,7195,3985,4257,5404,6852,3338,5795,6953,6688,5006,7316,5400,8339,8228,4587,5119,5514,3321,7005,5456,7182,4781,7260,5121,5463,8521,6032,8702,7188,5005,8818,4632,4910,7353,8946,4443,6754,4663,8363,4386,6658,8674,9084,6391,7714,6694,6039,7717,6807,7370,9091,5542,4328,7088,7031,4085,7236,6911,4569,7314,5043,7263,8569,7881,7874,6999,6566,7651,7095,7367,8452,8360,7947,6457,6727,6693,9089,7730,7047,8042,4279,4404,4599,8347,6655,6421,8831,8517,9050,8402,8930,8501,6677,7006,6880,7958,6666,7362,7085,8974,8016,7910,7185,6684,6023,9069,4311,8094,7937,8513,8975,6787,8976,8819,8959,7396,9033,4469,8003,7751,6945,3915,8428,4517,6724,7860,4616,6989,8713,7344,8064,6015,8540,9227,9162,9136,9318,9195,9223,9297,9206,9205,9331,9285,9194,9183,9309,9258,9267,9224]:
        print(f'{titles[movie]},')

    # all_movies = np.arange(len(df_train_matrix.columns))
    #
    # for slate_size in configs['slate_size']:
    #     set_seeds(configs['seed'])
    #     print(f'Test for {slate_size}')
    #     model = RandomSlateGeneration(slate_size, all_movies, configs['test_batch_size'])
    #
    #     experiment_builder = ExperimentBuilderRandom(model, test_loader, len(df_train_matrix.columns), movies_categories,
    #                                                  release_years, titles, configs)
    #     experiment_builder.run_experiment()


if __name__ == '__main__':
    experiments_run()
