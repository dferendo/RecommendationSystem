import argparse
import json
import os
import torch
import sys

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Welcome to the MLP course\'s Pytorch training and inference helper script')

    # Parameters for all experiments 

    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=False,
                        help='A flag indicating whether we will use GPU acceleration or not')

    parser.add_argument('--l2_regularizer', type=float, default=1e-5, help="l2 normalization constant")

    parser.add_argument('--on_cluster', type=str2bool,default = False, help="Flag to specify where the data will be held")                


    # Parameters for learning to rank approach

    parser.add_argument('--model', type=str, default="mf", help="mf/mlp/neuMF: Train baseline, either matrix factorization or neural network")    

    parser.add_argument('--dataset', type=str, default="1M", help="100K/1M/10M/20M")

    parser.add_argument('--experiment_name', type=str, default="matrix_model", help="Name of resulting experiment")    
    
    parser.add_argument('--precision_recall', type=str2bool, default=True, help="precision recall at k. If True the model will calculate it")    

    parser.add_argument('--map_recall', type=str2bool, default=True, help="mean average precision/  recall at k. If True the model will calculate it")    

    parser.add_argument('--rmse', type=str2bool, default=True, help="root mean square error. If True the model will calculate it")    
    
    parser.add_argument('--mf_embedding_dim', type=int, default=50, help="latents dimensions of matrix factorization models")

    parser.add_argument('--mlp_embedding_dim', type=int, default=16, help="latents dimensions of the embedding of mlp")
    
    parser.add_argument('--training_epochs', type=int, default=50, help="training epochs")

    parser.add_argument('--batch_size', type=int, default=20, help="training epochs")

    parser.add_argument('--learning_rate', type=float, default=0.00005, help=" learning rate")

    parser.add_argument('--optim', type=str, default="adam", help="adam/sgd/rms: optimizer to train the model")    

    parser.add_argument('--k', type=int, default=3, help="k:Variable to evaluate prec@k and rec@k")
    
    parser.add_argument('--neg_examples', type=int, default=5, help="number of negative examples per positive")


    # Parameters for learning to slate generation approach

    parser.add_argument('--optim_gan', type=str, default="adam", help="adam/sgd/rms: optimizer to train the model")

    parser.add_argument('--gan_embedding_dim', type=int, default=5, help="latents dimensions of matrix factorization models")
    
    parser.add_argument('--gan_hidden_layer', type=int, default=40, help="latents dimensions of matrix factorization models")
    
    parser.add_argument('--loss', type=str, default="bce", help="bce/mse: Error by which GANS are optimized")    

    parser.add_argument('--slate_size', type=int, default=3, help="Size of slate to be generated")
    parser.add_argument('--json_configs', nargs="?", type=str, default=None, help='')
    args = parser.parse_args()
    
    return args