import numpy as np

import scipy.stats as st

import torch 
import logging
from spotlight.torch_utils import cpu, gpu, minibatch, set_seed, shuffle
from spotlight.sampling import sample_items
FLOAT_MAX = np.finfo(np.float32).max



def mrr_score(model, test, train=None):
    """
    Compute mean reciprocal rank (MRR) scores. One score
    is given for every user with interactions in the test
    set, representing the mean reciprocal rank of all their
    test items.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, scores of known
        interactions will be set to very low values and so not
        affect the MRR.

    Returns
    -------

    mrr scores: numpy array of shape (num_users,)
        Array of MRR scores for each user in test.
    """

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    mrrs = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)

        if train is not None:
            predictions[train[user_id].indices] = FLOAT_MAX

        mrr = (1.0 / st.rankdata(predictions)[row.indices]).mean()

        mrrs.append(mrr)

    return np.array(mrrs)

def sequence_mrr_score(model, test, exclude_preceding=False):

    """
    Compute mean reciprocal rank (MRR) scores. Each sequence
    in test is split into two parts: the first part, containing
    all but the last elements, is used to predict the last element.

    The reciprocal rank of the last element is returned for each
    sequence.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.SequenceInteractions`
        Test interactions.
    exclude_preceding: boolean, optional
        When true, items already present in the sequence will
        be excluded from evaluation.

    Returns
    -------

    mrr scores: numpy array of shape (num_users,)
        Array of MRR scores for each sequence in test.
    """

    sequences = test.sequences[:, :-1]
    targets = test.sequences[:, -1:]

    mrrs = []

    for i in range(len(sequences)):

        predictions = -model.predict(sequences[i])

        if exclude_preceding:
            predictions[sequences[i]] = FLOAT_MAX

        mrr = (1.0 / st.rankdata(predictions)[targets[i]]).mean()

        mrrs.append(mrr)

    return np.array(mrrs)

def _get_precision_recall(predictions, targets, k):

    predictions = predictions[:k]
    num_hit = len(set(predictions).intersection(set(targets)))

    return float(num_hit) / k , float(num_hit) / len(targets)

def precision_recall_score(model, test, train=None, k=10):
    """
    Compute Precision@k and Recall@k scores. One score
    is given for every user with interactions in the test
    set, representing the Precision@k and Recall@k of all their
    test items.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, scores of known
        interactions will not affect the computed metrics.
    k: int or array of int,
        The maximum number of predicted items
    Returns
    -------

    (Precision@k, Recall@k): numpy array of shape (num_users, len(k))
        A tuple of Precisions@k and Recalls@k for each user in test.
        If k is a scalar, will return a tuple of vectors. If k is an
        array, will return a tuple of arrays, where each row corresponds
        to a user and each column corresponds to a value of k.
    """

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    if np.isscalar(k):
        k = np.array([k])

    precision = []
    recall = []
    cold_start_users=0
    for user_id,row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)

        if train is not None:
            rated = train[user_id].indices
            predictions[rated] = FLOAT_MAX
            if not len(rated):
                cold_start_users+=1
                # continue
        # Do differently for mlp
        predictions = predictions.argsort(axis=0)
        # targets = np.argwhere(row.toarray() >= threshold)[:, 1]

        targets = row.indices
        
        user_precision, user_recall = zip(*[
            _get_precision_recall(predictions, targets, x)
            for x in k
        ])

        precision.append(user_precision)
        recall.append(user_recall)

    precision = np.array(precision).squeeze()
    recall = np.array(recall).squeeze()
    print("Cold start users: ",cold_start_users)
    return np.mean(precision), np.mean(recall)

def rmse_score(net,user_ids,item_ids):
    predictions = net(user_ids, item_ids)
    array = 1 - predictions.cpu().detach().numpy()
    return np.sum(array**2)

def hit_ratio(model,test,k=10):
  
    test = test.tocsr()
   
    num_hits = 0 
    num_users = 0 
    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue
        
        predictions = -model.predict(user_id)
        predictions = predictions.argsort()
        # targets = np.argwhere(row.toarray() >= threshold)[:, 1]

        target = row.indices
        num_users+=1
        predictions = predictions[:k]
        if(target in predictions):
            num_hits+=1

    return num_hits/num_users

def evaluate_popItems(item_popularity, test,k=10):
    
    test = test.tocsr()
    
    pop_top = item_popularity.values.argsort()[::-1][:k]
    
    if np.isscalar(k):
        k = np.array([k])

    precision_popItem = []
    recall_popItem = []

    for row in test:

        if not len(row.indices):
            continue

        predictions = pop_top
        targets = row.indices

        user_precision, user_recall = zip(*[
            _get_precision_recall(predictions, targets, x)
            for x in k
        ])

        precision_popItem.append(user_precision)
        recall_popItem.append(user_recall)

    return np.mean(precision_popItem), np.mean(recall_popItem),

def evaluate_random(item_popularity, test,k=10):

    all_items = test.num_items
    
    test = test.tocsr()
    
    if np.isscalar(k):
        k = np.array([k])

    precision_random = []
    recall_random =[]
    no_movies = 0 
    for row in test:
        
        if not len(row.indices):
            no_movies+=1
            continue
        predictions = np.random.choice(all_items,len(row.indices))
        targets = row.indices

        user_precision, user_recall = zip(*[
            _get_precision_recall(predictions, targets, x)
            for x in k
        ])

        precision_random.append(user_precision)
        recall_random.append(user_recall)

    precision_random = np.array(precision_random).squeeze()
    recall_random = np.array(recall_random).squeeze()

    return np.mean(precision_random), np.mean(recall_random)

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual.any():
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def map_at_k(model,test,k = 5):
    
    test = test.tocsr()
    map_k_list = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)
        predictions = predictions.argsort()

        targets = row.indices
        apk_ = apk(targets,predictions,k=k)
        map_k_list.append(apk_)

    map_k= np.array(map_k_list).squeeze()

    return np.mean(map_k)

def precision_recall_score_slates(slates, test, k=3):

    
    # Delete items from test set that we don't have any training data
    if np.isscalar(k):
        k = np.array([k])
    
    precision = []
    recall = []

    #TODO: Very memoery intensive for big datasets
    
    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue
        # targets = np.argwhere(row.toarray() >= threshold)[:, 1]

        targets = row.indices
        user_precision, user_recall = zip(*[
            _get_precision_recall(slates[user_id].numpy(), targets, x)
            for x in k
        ])
        precision.append(user_precision[0])
        recall.append(user_recall[0])

   

    return precision, recall

def precision_recall_slates_atk(fake_slates,real_slates, k=3):
    
    # Delete items from test set that we don't have any training data
    if np.isscalar(k):
        k = np.array([k])
    
    precision = []
    recall = []
    # print(fake_slates[0,],real_slates[0,:])
    #TODO: Very memoery intensive for big datasets
    for user_id in range(fake_slates.shape[0]):

        targets = real_slates[user_id,:]
        predictions = fake_slates[user_id,:]
        user_precision, user_recall = zip(*[
            _get_precision_recall(predictions, targets, x)
            for x in k
        ])
        precision.append(user_precision[0])
        recall.append(user_recall[0])
   

    return precision, recall