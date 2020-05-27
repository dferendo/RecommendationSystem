from utils.arg_parser import extract_args_from_json
from utils.data_provider import split_dataset
from utils.reset_seed import set_seeds
from models import GreedyMLP
from dataloaders.PointwiseDataLoader import PointwiseDataLoader
from utils.experiment_builder import ExperimentBuilder

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch


class GreedyMLPExperimentBuilder(ExperimentBuilder):

    def pre_epoch_init_function(self):
        self.train_loader.dataset.negative_sampling()

    def loss_function(self, values):
        """
        Using the loss from the paper https://arxiv.org/pdf/1708.05031.pdf (ie Pointwise loss with negative sampling
        which is binary cross-entropy loss)
        """
        return torch.nn.BCELoss()

    def forward_model_training(self, values_to_unpack):
        user = values_to_unpack[0].cuda()
        positive_interaction_item = values_to_unpack[1].cuda()
        neg_sampled_item = values_to_unpack[2].cuda()

        prediction_i, prediction_j = self.model(user, positive_interaction_item, neg_sampled_item)

        return self.loss_function((prediction_i, prediction_j))


def experiments_run():
    configs = extract_args_from_json()
    set_seeds(configs['seed'])

    df_train, df_val, df_test, df_train_matrix = split_dataset(configs)

    train_dataset = PointwiseDataLoader(df_train, df_train_matrix, configs['negative_samples'], True)
    train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True, num_workers=4,
                              drop_last=True)

    model = GreedyMLP.GreedyMLP(len(df_train_matrix.index), len(df_train_matrix.columns), configs['hidden_layers_dims'],
                                configs['use_bias'])

    experiment_builder = GreedyMLPExperimentBuilder(model, train_loader, configs)
    experiment_builder.run_experiment()

    # optimizer = Adam(model.parameters(), amsgrad=False, weight_decay=1e-05)
    #
    # for epoch in range(configs['num_of_epochs']):
    #     model.train()
    #     # Get negative sampling
    #     train_loader.dataset.negative_sampling()
    #
    #     with tqdm.tqdm(total=len(train_loader), file=sys.stdout) as pbar_train:
    #         for user, positive_interaction_item, neg_sampled_item in train_loader:
    #             user = user.cuda()
    #             positive_interaction_item = positive_interaction_item.cuda()
    #             neg_sampled_item = neg_sampled_item.cuda()
    #
    #             model.zero_grad()
    #             prediction_i, prediction_j = model(user, positive_interaction_item, neg_sampled_item)
    #             loss = loss_function(prediction_i, prediction_j)
    #
    #             loss.backward()
    #             optimizer.step()
    #
    #             loss_value = loss.data.detach().cpu().numpy()
    #
    #             pbar_train.update(1)
    #             pbar_train.set_description("loss: {:.4f}".format(loss_value))
    #
    #     # log the epoch loss
    #     writer.add_scalar('training loss', loss_value, epoch)


if __name__ == '__main__':
    experiments_run()
