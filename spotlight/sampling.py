"""
Module containing functions for negative item sampling.
"""

import numpy as np
import logging
import time

def sample_items(interaction,user_ids,num_items, shape, random_state=None):
    """
    Randomly sample a number of items.

    Parameters
    ----------

    num_items: int
        Total number of items from which we should sample:
        the maximum value of a sampled item id will be smaller
        than this.
    shape: int or tuple of ints
        Shape of the sampled array.
    random_state: np.random.RandomState instance, optional
        Random state to use for sampling.

    Returns
    -------

    items: np.array of shape [shape]
        Sampled item ids.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    items = random_state.randint(0, num_items, shape, dtype=np.int64)

    return items

def negsamp_vectorized_bsearch_preverif(pos_inds, n_items, n_samp=32):
    """ Pre-verified with binary search
    `pos_inds` is assumed to be ordered
    """
    raw_samp = np.random.randint(0, n_items - len(pos_inds), size=n_samp)
    pos_inds_adj = pos_inds - np.arange(len(pos_inds))
    neg_inds = raw_samp + np.searchsorted(pos_inds_adj, raw_samp, side='right')
    return neg_inds

def get_negative_samples(train,num_samples):
    interactions = train.tocsr()

    num_items = train.num_items
    num_users = train.num_users
    
    logging.info("Generating %d Samples"%num_samples)
    start = time.time()

    users = np.random.choice(num_users,num_samples)
    items = np.random.choice(num_items,num_samples)

    neg_samples = []

    for i in range(num_samples):
        if train.has_key(users[i],items[i]):
            item = negsamp_vectorized_bsearch_preverif (interactions[users[i],:].toarray().nonzero()[1],num_items,1)[0]
            pair = (users[i],item)
        else:
            pair = (users[i],items[i])
        neg_samples.append(pair)

    end = time.time()
    logging.info("Took %d seconds"%(end - start))
    return neg_samples


