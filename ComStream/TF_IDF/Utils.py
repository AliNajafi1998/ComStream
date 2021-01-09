import copy
from math import sqrt, log


def get_distance_tf_idf_cosine(king_agent, freq1: dict, freq2: dict):
    """
    get the distance using cosine similarity with tf-idf
    :param king_agent: the object of the king agent so we have the global variables
    :param freq1: the frequencies of the first dictionary of {token_id:frequencies}
    :param freq2: the frequencies of the second dictionary of {token_id:frequencies}
    :return: (int) returns the distance of the 2 frequency dictionaries
    """
    idf = calculate_idf(king_agent, freq1, freq2)

    sum_freq1 = sum(freq1.values())
    sum_freq2 = sum(freq2.values())

    sum_product_dots = 0.0
    dp1_tf_idf_length = 0.0
    dp2_tf_idf_length = 0.0
    for token_id_in_dp1 in freq1:
        if token_id_in_dp1 in freq2:
            sum_product_dots += (freq1[token_id_in_dp1] / sum_freq1 * idf[token_id_in_dp1]) * (
                    freq2[token_id_in_dp1] / sum_freq2 * idf[token_id_in_dp1])
        dp1_tf_idf_length += pow((freq1[token_id_in_dp1] / sum_freq1) * idf[token_id_in_dp1], 2)

    dp1_tf_idf_length = sqrt(dp1_tf_idf_length)

    for token_id_in_dp2 in freq2:
        dp2_tf_idf_length += pow((freq2[token_id_in_dp2] / sum_freq2) * idf[token_id_in_dp2], 2)
    dp2_tf_idf_length = sqrt(dp2_tf_idf_length)
    if dp1_tf_idf_length < king_agent.data_agent.epsilon or dp2_tf_idf_length < king_agent.data_agent.epsilon:
        return 1.0
    else:
        return 1.0 - (sum_product_dots / ((dp1_tf_idf_length * dp2_tf_idf_length) + king_agent.data_agent.epsilon))


def calculate_idf(king_agent, freq1: dict, freq2: dict):
    """
    calculates the idf for each token and returns it
    :param king_agent: the object of the king agent so we have the global variables
    :param freq1: the frequencies of the first dictionary of {token_id:frequencies}
    :param freq2: the frequencies of the second dictionary of {token_id:frequencies}
    :return: returns a dictionary of idfs for each token {token_id:frequencies}
    """
    idf_dictionary = dict()
    freq = list(freq1.keys())
    freq.extend(list(freq2.keys()))
    freq = set(freq)

    for token_id in freq:
        counter = 1 + king_agent.global_idf_count.get(token_id, 0)
        num = 1 + len(king_agent.agents)
        idf_dictionary[token_id] = 1 + log(num / counter)

    return idf_dictionary


def get_seconds(time: str):
    """
    calculates the seconds of a time
    :param time: the time as str 'hh:mm:ss'
    :return: returns time as seconds
    """
    s = time.split(':')
    seconds = float(s[2]) + float(s[1]) * 60 + float(s[0]) * 3600
    return seconds
