import torch
import numpy as np
import logging
import spotlight.optimizers as optimizers
import tqdm
from utils.arg_parser import extract_args_from_json

from utils.different import CGAN
from utils.arg_extractor import get_args
from utils.slate_formation import get_data_loaders

from spotlight.dnn_models.cGAN_models import generator, discriminator
from spotlight.torch_utils import cpu, gpu, minibatch, set_seed, shuffle

logging.basicConfig(format='%(message)s',level=logging.INFO)

args = get_args()  # get arguments from command line
use_cuda=args.use_gpu
dataset_name = args.dataset

logging.info("DataSet MovieLens_%s will be used"%dataset_name)

if args.on_cluster:
    path = '/disk/scratch/s1877727/datasets/movielens/'
else:
    path = 'datasets/movielens/'


from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype


def get_sparse_df(df, all_movies_in_train):
    users_category = CategoricalDtype(sorted(df['userId'].unique()), ordered=True)
    movies_category = CategoricalDtype(sorted(all_movies_in_train), ordered=True)

    row = df['userId'].astype(users_category).cat.codes
    col = df['movieId'].astype(movies_category).cat.codes

    sparse_matrix = csr_matrix((df["rating"], (row, col)),
                               shape=(users_category.categories.size, movies_category.categories.size))

    return sparse_matrix


#Reproducability of results
configs = extract_args_from_json()
print(configs)
seed = 0
random_state = np.random.RandomState(seed)
torch.manual_seed(seed)

# Read required arguments from user inputs
total_movies = -1
min_movies = 0
min_viewers = 5

train_loader, test_loader, data_configs, df_test, df_train = get_data_loaders(configs, True)

train_vec = torch.from_numpy(train_loader.dataset.padded_interactions)
train_slates = train_loader.dataset.slate_vector_matrix

num_users = data_configs['number_of_users']
num_movies = data_configs['number_of_movies']

valid_vec = torch.from_numpy(test_loader.dataset.padded_interactions)
valid_set = get_sparse_df(df_test, df_train['movieId'].unique())

# In general should be smaller than the dimensions of the output (Latent dimensions < Visible dimensions)
noise_dim = 50

Gen = generator(num_items = num_movies, noise_dim = noise_dim,
                embedding_dim = args.gan_embedding_dim,
                hidden_layer = [args.gan_hidden_layer//2,args.gan_hidden_layer],
                output_dim=args.slate_size )

Disc = discriminator(num_items= num_movies,
                     embedding_dim = args.gan_embedding_dim,
                     hidden_layers = [2*args.gan_hidden_layer,args.gan_hidden_layer,args.gan_hidden_layer//2],
                     input_dim=args.slate_size )



''' Shallow architectures of our generator and discriminator 

Gen = generator(num_items = num_movies, noise_dim = noise_dim, 
                embedding_dim = args.gan_embedding_dim, 
                hidden_layer = [args.gan_hidden_layer], 
                output_dim=args.slate_size )

Disc = discriminator(num_items= num_movies, 
                     embedding_dim = args.gan_embedding_dim, 
                     hidden_layers = [2*args.gan_hidden_layer], 
                     input_dim=args.slate_size )
'''

# Choose optimizer for both generator and discriminator

optim = getattr(optimizers, args.optim_gan + '_optimizer')

model = CGAN(   n_iter=args.training_epochs,
                z_dim = noise_dim,
                embedding_dim = args.gan_embedding_dim,
                hidden_layer = args.gan_hidden_layer,
                batch_size=args.batch_size,
                loss_fun = args.loss,
                slate_size = args.slate_size ,
                learning_rate=args.learning_rate,
                use_cuda=use_cuda,
                experiment_name=args.experiment_name,
                G_optimizer_func = optim,
                D_optimizer_func = optim,
                G=Gen,
                D=Disc
                )


logging.info(" Training session: {}  epochs, {} batch size {} learning rate.  {} users x  {} items"
                                    .format(args.training_epochs, args.batch_size,
                                      args.learning_rate,num_users, num_movies)
            )

logging.info("Model set, training begins")
model.fit(train_vec,train_slates, num_users, num_movies, valid_vec, None, valid_set)
logging.info("Model is ready, testing performance")
