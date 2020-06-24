import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,layers,num_users, num_items,output_dim = 1, embedding_dim=32):
        super(MLP, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = embedding_dim

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # self.affine_output = torch.nn.Linear(layers[-1], out_features=1)

        self.layers = nn.ModuleList()
        
        for idx in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[idx],layers[idx+1]))
            # self.layers.append(nn.BatchNorm1d(num_features=mlp_layers[idx+1]))
            self.layers.append(nn.LeakyReLU(0.1,inplace=True))
            self.layers.append(nn.Dropout(0.5))
            
        self.layers.append(torch.nn.Linear(layers[-1], out_features=1))
        self.logistic = torch.nn.Sigmoid()
        self.apply(self.init_weights)

    def forward(self, user_indices, item_indices):

        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)

        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        
        for layers in self.layers[:-1]:
            vector = layers(vector)
        logits =  self.layers[-1](vector)
        rating = self.logistic(logits)
        return rating

    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
