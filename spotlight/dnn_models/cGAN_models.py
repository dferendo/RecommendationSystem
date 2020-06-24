import torch
import torch.nn as nn
import numpy as np
from spotlight.torch_utils import cpu, gpu, minibatch, set_seed, shuffle

class parameter_learning(nn.Module):
    def __init__(self):
        super(parameter_learning, self).__init__()  


class generator(nn.Module):
    def __init__(self, noise_dim = 100,embedding_dim = 50, hidden_layer = [16], num_items=1447,output_dim = 3):

        super(generator, self).__init__()  

        self.z = noise_dim
        self.y = embedding_dim
        self.num_items = num_items
        self.output_dim = output_dim

        self.embedding_layer = nn.Embedding( self.num_items+1, self.y, padding_idx=self.num_items)
        
        # List to store the dimensions of the layers
        self.layers = nn.ModuleList()
        layers = [self.z + self.y] + hidden_layer
        
        for idx in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[idx], layers[idx+1]))
            self.layers.append(nn.BatchNorm1d(num_features=layers[idx+1]))
            self.layers.append(nn.Dropout(0.2))
            self.layers.append(nn.LeakyReLU(0.01,inplace=True))
        
        self.mult_heads =  nn.ModuleDict({})
        for b in range(self.output_dim):
            self.mult_heads['head_'+str(b)] =  nn.Linear(layers[-1], self.num_items )

        self.apply(self.init_weights)
        self.non_linear_emb = nn.LeakyReLU(0.02,inplace=True)


    def forward(self, noise, user_batch,inference=False):

        # Returns multiple exits, one for each item.
        raw_emb = self.embedding_layer(user_batch.long())
        user_emb = raw_emb.sum(1)
        vector = torch.cat([noise, user_emb], dim=1)
        vector = self.non_linear_emb(vector)
        for layers in self.layers:
            vector = layers(vector)
        
        outputs_tensors = []
        if inference:
            # Return the item in int format to suggest items to users
            for output in self.mult_heads.values():
                out = output(vector)
                out = torch.tanh(out)
                _,indices = torch.max(out,1)
                outputs_tensors.append(indices)
            slates = torch.empty([noise.shape[0],len(self.mult_heads)])
            for i,items in enumerate(zip(*tuple(outputs_tensors))):
                slates[i,:] = torch.stack(items)
            return slates
        else:
            for output in self.mult_heads.values():
                out = output(vector)
                out = torch.tanh(out)
                outputs_tensors.append(out)
            return tuple(outputs_tensors)#, user_emb
            
    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
class discriminator(nn.Module):
    def __init__(self, embedding_dim = 50 ,  hidden_layers = [16], input_dim = 3, num_items=1447):
        super(discriminator, self).__init__()

        self.non_linear_emb = nn.LeakyReLU(0.2,inplace=True)

        self.slate_size = input_dim
        self.user_condition = embedding_dim
        self.num_items = num_items
        self.embedding_layer = nn.Embedding(  self.num_items+1,
                                        self.user_condition,
                                        padding_idx=self.num_items)

        #List to store the dimensions of the layers
        self.layers =  nn.ModuleList()
        layers = [self.slate_size*self.num_items + self.user_condition] + hidden_layers + [1]

        for idx in range(len(layers)-2):
            self.layers.append(nn.Linear(layers[idx], layers[idx+1]))
            self.layers.append(nn.Dropout(0.2))
            self.layers.append(nn.LeakyReLU(0.02))
        
        self.layers.append(nn.Linear(layers[-2], layers[-1]))
        self.apply(self.init_weights)

        

    def forward(self, batch_input, condition):#,user_emb):
        raw_emb = self.embedding_layer(condition.long())
        user_emb = raw_emb.sum(1)
        vector =  self.non_linear_emb(torch.cat([user_emb, batch_input], dim=1).float()) # the concat latent vector
        vector = torch.cat([user_emb, batch_input], dim=1).float() # the concat latent vector
        for layers in self.layers:
            vector = layers(vector)
        return vector

    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


