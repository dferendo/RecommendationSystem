import torch
import torch.nn as nn
from spotlight.factorization.representations import BilinearNet as GMF
from spotlight.dnn_models.mlp import MLP


class NeuMF(nn.Module):
    def __init__(self, mlp_layers,num_users, num_items,mf_embedding_dim=25, mlp_embedding_dim=32):
        super(NeuMF, self).__init__()
        # self.config = config
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim_mf = mf_embedding_dim
        self.latent_dim_mlp = mlp_embedding_dim
        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf )
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        # Initialize the neural network
        
        self.layers = nn.ModuleList()
        
        for idx in range(len(mlp_layers)-1):
            self.layers.append(nn.Linear(mlp_layers[idx],mlp_layers[idx+1]))
            self.layers.append(nn.LeakyReLU(0.1,inplace=True))
            self.layers.append(nn.Dropout(0.5))
            
            
        self.affine_output = torch.nn.Linear(mlp_layers[-1]+mf_embedding_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()
        self.apply(self.init_weights)

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        
        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)

        for layers in self.layers:
            mlp_vector = layers(mlp_vector)
            # mlp_vector = torch.tanh(mlp_vector)
        
        vector = torch.cat([mlp_vector, mf_vector],dim=-1)


        logits = self.affine_output(vector)
        rating = self.logistic(logits)

        return rating


    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    

   
