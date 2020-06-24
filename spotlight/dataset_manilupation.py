"""
Module with functionality for splitting and shuffling datasets.
"""

import numpy as np

from sklearn.utils import murmurhash3_32
from scipy.sparse import coo_matrix,csr_matrix
from spotlight.interactions import Interactions



def _index_or_none(array, shuffle_index):

    if array is None:
        return None
    else:
        return array[shuffle_index]

def shuffle_interactions(interactions,
                         random_state=None):
    """
    Shuffle interactions.

    Parameters
    ----------

    interactions: :class:`spotlight.interactions.Interactions`
        The interactions to shuffle.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.

    Returns
    -------

    interactions: :class:`spotlight.interactions.Interactions`
        The shuffled interactions.
    """

    if random_state is None:
        random_state = np.random.RandomState()

    shuffle_indices = np.arange(len(interactions.user_ids))
    random_state.shuffle(shuffle_indices)

    return Interactions(interactions.user_ids[shuffle_indices],
                        interactions.item_ids[shuffle_indices],
                        ratings=_index_or_none(interactions.ratings,
                                               shuffle_indices),
                        timestamps=_index_or_none(interactions.timestamps,
                                                  shuffle_indices),
                        weights=_index_or_none(interactions.weights,
                                               shuffle_indices),
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)

def random_train_test_split(interactions,
                            test_percentage=0.2,
                            random_state=None):
    """
    Randomly split interactions between training and testing.

    Parameters
    ----------

    interactions: :class:`spotlight.interactions.Interactions`
        The interactions to shuffle.
    test_percentage: float, optional
        The fraction of interactions to place in the test set.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.

    Returns
    -------

    (train, test): (:class:`spotlight.interactions.Interactions`,
                    :class:`spotlight.interactions.Interactions`)
         A tuple of (train data, test data)
    """

    interactions = shuffle_interactions(interactions,
                                        random_state=random_state)

    cutoff = int((1.0 - test_percentage) * len(interactions))

    train_idx = slice(None, cutoff)
    test_idx = slice(cutoff, None)

    train = Interactions(interactions.user_ids[train_idx],
                         interactions.item_ids[train_idx],
                         ratings=_index_or_none(interactions.ratings,
                                                train_idx),
                         timestamps=_index_or_none(interactions.timestamps,
                                                   train_idx),
                         weights=_index_or_none(interactions.weights,
                                                train_idx),
                         num_users=interactions.num_users,
                         num_items=interactions.num_items)

    test = Interactions(interactions.user_ids[test_idx],
                        interactions.item_ids[test_idx],
                        ratings=_index_or_none(interactions.ratings,
                                               test_idx),
                        timestamps=_index_or_none(interactions.timestamps,
                                                  test_idx),
                        weights=_index_or_none(interactions.weights,
                                               test_idx),
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)

    return train, test

def user_based_train_test_split(interactions,
                                test_percentage=0.2,
                                random_state=None):
    """
    Split interactions between a train and a test set based on
    user ids, so that a given user's entire interaction history
    is either in the train, or the test set.

    Parameters
    ----------

    interactions: :class:`spotlight.interactions.Interactions`
        The interactions to shuffle.
    test_percentage: float, optional
        The fraction of users to place in the test set.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.

    Returns
    -------

    (train, test): (:class:`spotlight.interactions.Interactions`,
                    :class:`spotlight.interactions.Interactions`)
         A tuple of (train data, test data)
    """

    if random_state is None:
        random_state = np.random.RandomState()

    minint = np.iinfo(np.uint32).min
    maxint = np.iinfo(np.uint32).max

    seed = random_state.randint(minint, maxint, dtype=np.int64)

    in_test = ((murmurhash3_32(interactions.user_ids,
                               seed=seed,
                               positive=True) % 100 /
                100.0) <
               test_percentage)
    in_train = np.logical_not(in_test)

    train = Interactions(interactions.user_ids[in_train],
                         interactions.item_ids[in_train],
                         ratings=_index_or_none(interactions.ratings,
                                                in_train),
                         timestamps=_index_or_none(interactions.timestamps,
                                                   in_train),
                         weights=_index_or_none(interactions.weights,
                                                in_train),
                         num_users=interactions.num_users,
                         num_items=interactions.num_items)
    test = Interactions(interactions.user_ids[in_test],
                        interactions.item_ids[in_test],
                        ratings=_index_or_none(interactions.ratings,
                                               in_test),
                        timestamps=_index_or_none(interactions.timestamps,
                                                  in_test),
                        weights=_index_or_none(interactions.weights,
                                               in_test),
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)

    return train, test
    
def train_test_timebased_split(interactions, test_percentage=0.2):
                            
    """
    Split interactions between training and testing, based on the timestamp.
    Interaction are split into interaction before a certain timestamp (test_percentage) and after

    Parameters
    ----------

    interactions: :class:`spotlight.interactions.Interactions`
        The interactions array
    test_percentage: float, optional
        The fraction of interactions to place in the test set.

    Returns
    -------

    (train, test): (:class:`spotlight.interactions.Interactions`,
                    :class:`spotlight.interactions.Interactions`)
         A tuple of (train data, test data)
    """

    user = interactions.user_ids
    items = interactions.item_ids
    timestamps = interactions.timestamps
    
    # Sort all items 
    index = timestamps.argsort()
    
    interactions.user_ids = user[index]
    interactions.item_ids = items[index]
    interactions.timestamps = timestamps[index]

    cutoff = int((1.0 - test_percentage) * len(interactions))

    train_idx = slice(None, cutoff)
    test_idx = slice(cutoff, None)
    train = Interactions(interactions.user_ids[train_idx],
                         interactions.item_ids[train_idx],
                         ratings=_index_or_none(interactions.ratings,
                                                train_idx),
                         timestamps=_index_or_none(interactions.timestamps,
                                                   train_idx),
                         weights=_index_or_none(interactions.weights,
                                                train_idx),
                         num_users=interactions.num_users,
                         num_items=interactions.num_items)

    test = Interactions(interactions.user_ids[test_idx],
                        interactions.item_ids[test_idx],
                        ratings=_index_or_none(interactions.ratings,
                                               test_idx),
                        timestamps=_index_or_none(interactions.timestamps,
                                                  test_idx),
                        weights=_index_or_none(interactions.weights,
                                               test_idx),
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)

    return train, test

def delete_rows_csr(mat, row_indices=[], col_indices=[]):
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:,col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:,mask]
    else:
        return mat

def create_slates(interactions,n=5,padding_value=0):
    
    """
    
    Given a user,movies,rating Interactions class and returns 
    for every user his last n-intercations as a list.

    Removes the interactions from the training set. Users that have no more than 5 interactions are also removed
    ----------

        Interactions: class:`spotlight.interactions.Interactions`
        n: int, number of last interactions
        Returns

    ----------

        interactions :  class:`spotlight.interactions.Interactions`
        slates = np.array, shape = (num_users, n)
    
    """
    num_users = interactions.num_users
    slates = np.zeros((num_users,n))
    indexes_to_delete = []
    for user_id in range(num_users):
        indices = np.where(interactions.user_ids == user_id)
        if len(indices[0]) == 0:
            continue
        elif len(indices[0]) < n :
            indexes_to_delete += list(indices[0])
            continue
        timestamps_sorted = interactions.timestamps[indices].argsort()
        indexes_sorted = indices[0][timestamps_sorted]
        slates[user_id] = interactions.item_ids[indexes_sorted[-n:]]
        indexes_to_delete += list(indexes_sorted[-n:])
    
    interactions.user_ids   = np.delete(interactions.user_ids,indexes_to_delete)
    interactions.item_ids   = np.delete(interactions.item_ids,indexes_to_delete)
    interactions.timestamps = np.delete(interactions.timestamps,indexes_to_delete)
    interactions.ratings    = np.delete(interactions.ratings,indexes_to_delete)

    # Remove users that do not have more than 5 interactions
    zero_indices = np.where(~slates.any(axis=1))[0]
    
    slates = np.delete(slates,zero_indices,axis=0)
    csr_mtrx = interactions.tocsr()
    interactions = delete_rows_csr(csr_mtrx,row_indices = list(zero_indices))
    return interactions,slates

def train_test_split(interactions,test_percentage=0.2):
    
    """
    
    Split an interactions matrix into training and test sets.
    Parameters
    ----------
    interactions : np.ndarray
    n : int (default=10)
        Number of items to select / row to place into test.
    Returns
    -------
    train : np.ndarray
    test : np.ndarray
    
    """
    
    interactions = interactions.tocsr().todense()
    test = np.zeros(interactions.shape)
    train = interactions.copy()

    for user in range(interactions.shape[0]):
        test_interactions = int(interactions[user, :].nonzero()[1].shape[0]*test_percentage)
        test_interactions = np.random.choice(interactions[user, :].nonzero()[1],test_interactions)
        train[user, test_interactions] = 0
        test[user, test_interactions] = interactions[user, test_interactions]

    # Test and training are truly disjoint
    assert(np.all(np.multiply(train,test) == 0))

    train_coo = coo_matrix(train)
    test_coo = coo_matrix(test)
    train_inter = Interactions(train_coo.row,
                         train_coo.col,
                         train_coo.data,
                         timestamps=None,
                         weights=None,
                         num_users=interactions.shape[0],
                         num_items=interactions.shape[1])
    test_inter = Interactions(test_coo.row,
                         test_coo.col,
                         test_coo.data,
                         timestamps=None,
                         weights=None,
                        num_users=interactions.shape[0],
                         num_items=interactions.shape[1])
    return train_inter, test_inter
