import numpy as np


def get_seconds(time: str):
    """
    calculates the seconds of a time
    :param time: the time as str 'hh:mm:ss'
    :return: returns time as seconds
    """
    s = time.split(':')
    seconds = float(s[2]) + float(s[1]) * 60 + float(s[0]) * 3600
    return seconds


def get_distance_cosine(vec_1, vec_2):
    return (1 - (np.dot(vec_1, vec_2.T)) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))) / 2.0


def get_distance_euclidean(vec_1, vec_2):
    return np.sqrt(np.sum((vec_1 - vec_2) ** 2))
