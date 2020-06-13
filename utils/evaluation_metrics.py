import numpy as np
import torch


def precision_hit_ratio(predicted_slates, ground_truths):
    recall_score = 0.0
    precision_score = 0.0

    for predicted_slate, ground_truths_slate in list(zip(predicted_slates, ground_truths)):
        intersection = np.intersect1d(predicted_slate, ground_truths_slate)

        recall_score += (len(intersection) / (1.0 * len(ground_truths_slate)))
        precision_score += (len(intersection) / (1.0 * len(predicted_slate)))

    return precision_score / (1.0 * predicted_slates.size(0)), recall_score / (1.0 * predicted_slates.size(0))


def movie_diversity(predicted_slates, all_movies_count):
    """
    :param predicted_slates:
    :param all_movies_count:
    :return:
    """
    return (torch.unique(predicted_slates).size()[0] * 1.0) / all_movies_count
