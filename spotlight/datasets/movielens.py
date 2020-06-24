"""
Utilities for fetching the Movielens datasets [1]_.

References
----------

.. [1] https://grouplens.org/datasets/movielens/
"""

import os
import pandas as pd
import h5py
import numpy as np
import torch 

import logging

logging.basicConfig(format='%(message)s',level=logging.INFO)

from spotlight.datasets import _transport
from spotlight.interactions import Interactions

VARIANTS = ('100K',
            '1M',
            '10M',
            '20M')


URL_PREFIX = ('https://github.com/maciejkula/recommender_datasets/'
              'releases/download')
VERSION = 'v0.2.0'


def _get_movielens(dataset):

    """
        Responsible for fetching the desired datasets from memory

        Parameters
        -------------
          dataset: relative or absolute path to the desired dataset. Options are : {100K,1M,10M,20M}

        Output
        -----------
        data: tuple of np.arrays containing user_id, item_id, rating, timestamp

    """

    extension = '.hdf5'
    path =  dataset + extension

    logging.info("Data will be read from file: " + path)

    with h5py.File(path, 'r') as data:
        return (data['/user_id'][:],
                data['/item_id'][:],
                data['/rating'][:],
                data['/timestamp'][:])
   
def keep_top_k(dataset, k):
    valid_movies = dataset.groupby('movieId',as_index=False).size().sort_values(ascending = False)[:1000].index.values
    return dataset.loc[dataset['movieId'].isin(valid_movies)]
    
             

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    # Only keep the triplets for items which were clicked on by at least min_sc users. 
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]
    
    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]
    
    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId') 
    return tp, usercount, itemcount
    

def get_movielens_dataset(variant='100K',path=None,min_uc=5, min_sc=0, movies_to_keep = -1):

    """
    Download and return one of the Movielens datasets.

    Parameters
    ----------

    variant: string, optional
         String specifying which of the Movielens datasets
         to download. One of ('100K', '1M', '10M', '20M').

    Returns
    -------

    Interactions: :class:`spotlight.interactions.Interactions`
        instance of the interactions class

    """

    if variant not in VARIANTS:
        raise ValueError('Variant must be one of {}, '
                         'got {}.'.format(VARIANTS, variant))
    url = 'movielens_{}'.format(variant)
    if path:
        url = path+url
    users,items,ratings,timestamps =(_get_movielens(url))

    dataset = pd.DataFrame({'userId':users,'movieId':items,'rating':ratings,'timestamps':timestamps})
    
    dataset = dataset[dataset['rating'] > 3.5]

    dataset, _, itemcount = filter_triplets(dataset, min_uc=min_uc, min_sc=min_sc)
    if movies_to_keep != -1:
        print(movies_to_keep)
        print(itemcount.size)
        if movies_to_keep < itemcount.size:
            dataset = keep_top_k(dataset, movies_to_keep)


    num_users = len(dataset.userId.unique())
    num_items = len(dataset.movieId.unique())
    logging.info("{} users and {} items".format(num_users,num_items))
    
    users,items,ratings,timestamps = dataset.userId.values,dataset.movieId.values,dataset.rating.values,dataset.timestamps.values


    user_to_id = dict((sid, i) for (i, sid) in enumerate(dataset.userId.unique()))
    item_to_id = dict((pid, i) for (i, pid) in enumerate(dataset.movieId.unique()))

    uid = np.array(list(map(lambda x: user_to_id[x], users)))
    sid = np.array(list(map(lambda x: item_to_id[x], items)))

    return Interactions(uid,sid,ratings,timestamps,num_users=num_users,num_items=num_items),itemcount
