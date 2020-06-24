import torch, time, os, pickle
import torch.nn as nn

import numpy as np
import torch
import torch.optim as optim
import random
import logging
import tqdm
import copy

from spotlight.helpers import _repr_model
from spotlight.factorization._components import _predict_process_ids
from spotlight.dataset_manilupation import create_user_embedding
from spotlight.torch_utils import cpu, gpu, minibatch, set_seed, shuffle
from spotlight.evaluation import rmse_score,precision_recall_score,evaluate_popItems,evaluate_random,hit_ratio


class cdae(nn.Module):
    def __init__(self,hidden_neurons, num_users,num_items,corruption_level,user_embdedding_dim =32):

        self.num_users = num_users
        self.num_items = num_items
        self.corruption_level = corruption_level
        self.hidden_neurons = hidden_neurons
        self.user_embdedding_dim = user_embdedding_dim


        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=hidden_neurons)
        self.fc1 = nn.Linear(num_items+user_embdedding_dim, hidden_neurons,bias=True)
        self.fc2 = nn.Linear(hidden_neurons,num_items,bias=True)

        self.activation_to_hidden = nn.functional.relu()
        self.activation_from_hidden = nn.functioncal.relu()


    def forward(self,item_input_nodes,user_indices):

        user_embdedding = self.embedding_user(user_indices)
        vector = torch.cat([user_embdedding,item_input_nodes],dim=-1)
        vector = self.fc1(vector)
        vector = self.activation_to_hidden(vector)
        vector = self.fc2(vector)
        vector = self.activation_from_hidden(vector)

        return vector
        
    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)