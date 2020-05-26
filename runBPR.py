from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset
from utils.reset_seed import set_seeds
from models import BayesianPR
from dataloaders.NegativeSamplingDataLoader import NSData

import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.optim.adam import Adam

import tqdm
import sys


def loss_function(prediction_i, prediction_j):
    """
    Using the loss from the paper https://arxiv.org/pdf/1205.2618.pdf
    """
    return ((prediction_i - prediction_j).sigmoid().log().sum()) * -1


def experiments_run():
    configs = extract_args_from_json()
    set_seeds(configs['seed'])

    df_train, df_val, df_test, df_train_matrix = split_dataset(configs)

    train_dataset = NSData(df_train, df_train_matrix, configs['negative_samples'], True)
    train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True, num_workers=4)

    model = BayesianPR.BPR(len(df_train_matrix.index), len(df_train_matrix.columns), configs['hidden_dims'])
    model.reset_parameters()

    if torch.cuda.device_count() > 1 and configs['use_gpu']:
        device = torch.cuda.current_device()
        model.to(device)
        model = nn.DataParallel(module=model)
        print('Use Multi GPU', device)
    elif torch.cuda.device_count() == 1 and configs['use_gpu']:
        device = torch.cuda.current_device()
        model.to(device)  # sends the model from the cpu to the gpu
        print('Use GPU', device)
    else:
        print("use CPU")
        device = torch.device('cpu')  # sets the device to be CPU
        print(device)

    model.cuda()

    optimizer = Adam(model.parameters(), amsgrad=False, weight_decay=1e-05)

    for epoch in range(configs['num_of_epochs']):
        model.train()
        # Get negative sampling
        train_loader.dataset.negative_sampling()

        with tqdm.tqdm(total=len(train_loader), file=sys.stdout) as pbar_train:
            for user, positive_interaction_item, neg_sampled_item in train_loader:
                user = user.cuda()
                positive_interaction_item = positive_interaction_item.cuda()
                neg_sampled_item = neg_sampled_item.cuda()

                model.zero_grad()
                prediction_i, prediction_j = model(user, positive_interaction_item, neg_sampled_item)
                loss = loss_function(prediction_i, prediction_j)

                loss.backward()
                optimizer.step()

                loss_value = loss.data.detach().cpu().numpy()

                pbar_train.update(1)
                pbar_train.set_description("loss: {:.4f}".format(loss))


if __name__ == '__main__':
    experiments_run()
