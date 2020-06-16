import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GreedyMLP(nn.Module):
    """
    Implementation follows the model proposed by (Neural Collaborative Filtering) https://arxiv.org/pdf/1708.05031.pdf
    We will use a pointwise loss function. Note this method is also known as DeepRank.
    """
    def __init__(self, num_users, num_items, hidden_layers_dim, use_bias, dropout=0):
        super(GreedyMLP, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.hidden_layers_dim = hidden_layers_dim
        self.use_bias = use_bias
        self.dropout = dropout

        self.layer_dict = nn.ModuleDict()

        # Embedding, hidden_layers_dim[0] represents the hidden dims for both embeddings
        latent_dims = self.hidden_layers_dim[0] // 2

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=latent_dims)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=latent_dims)

        # Pairwise iteration [ie output_size will be the input_size of the next iteration]
        for idx, (input_size, output_size) in enumerate(zip(self.hidden_layers_dim, self.hidden_layers_dim[1:])):
            self.layer_dict[f'fcc_{idx}'] = nn.Linear(in_features=input_size, out_features=output_size,
                                                      bias=self.use_bias)
            self.layer_dict[f'bn_{idx}'] = nn.BatchNorm1d(output_size)

            if self.dropout != 0:
                self.layer_dict[f'dropout_{idx}'] = nn.Dropout(p=self.dropout)

        # Output layer (Binary classification since we are doing pointwise loss)
        self.classifier = torch.nn.Linear(in_features=self.hidden_layers_dim[-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)

        out = torch.cat([user_embedding, item_embedding], dim=-1)

        # -1 to remove the embedding layer
        for idx in range(len(self.hidden_layers_dim) - 1):
            out = self.layer_dict[f'fcc_{idx}'](out)
            out = F.relu_(out)
            out = self.layer_dict[f'bn_{idx}'](out)

            if self.dropout != 0:
                out = self.layer_dict[f'dropout_{idx}'](out)

        out = self.classifier(out)
        out = self.logistic(out)
        
        return out

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        self.embedding_user.reset_parameters()
        self.embedding_item.reset_parameters()

        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except AttributeError:
                pass

        self.classifier.reset_parameters()
