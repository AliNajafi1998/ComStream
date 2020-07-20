from math import sqrt, log
from .DataAgent import DataAgent


def get_distance_tf_idf_cosine(data_agent: DataAgent, tf1: dict, tf2: dict):
    idf = calculate_idf(data_agent, tf1, tf2)

    sum_product_dots = 0.0
    dp1_tf_idf_length = 0.0
    dp2_tf_idf_length = 0.0
    for term_in_dp1 in tf1:
        if term_in_dp1 in idf:
            if term_in_dp1 in tf2:
                sum_product_dots += (tf1[term_in_dp1] * idf[term_in_dp1]) * (tf2[term_in_dp1] * idf[term_in_dp1])
            dp1_tf_idf_length += pow(tf1[term_in_dp1] * idf[term_in_dp1], 2)

    dp1_tf_idf_length = sqrt(dp1_tf_idf_length)

    for term_in_dp2 in tf2:
        if term_in_dp2 in idf:
            dp2_tf_idf_length += pow(tf2[term_in_dp2] * idf[term_in_dp2], 2)
    dp2_tf_idf_length = sqrt(dp2_tf_idf_length)

    if dp1_tf_idf_length < data_agent.epsilon or dp2_tf_idf_length < data_agent.epsilon:
        return 1.0
    else:
        return 1.0 - (sum_product_dots / (dp1_tf_idf_length * dp2_tf_idf_length))


def calculate_idf(data_agent: DataAgent, tf1: dict, tf2: dict):
    terms_global_frequency = data_agent.terms_global_frequency
    idf_dictionary = dict()

    for term in tf1:
        if data_agent.global_tf[term] < data_agent.epsilon:
            idf_dictionary[term] = 0.0
        else:
            idf_dictionary[term] = 1.0 + log(terms_global_frequency / data_agent.global_tf[term])

    for term in tf2:
        if term not in idf_dictionary:
            if data_agent.global_tf[term] < data_agent.epsilon:
                idf_dictionary[term] = 0.0
            else:
                idf_dictionary[term] = 1.0 + log(terms_global_frequency / data_agent.global_tf[term])

    return idf_dictionary


def get_seconds(time: str):
    s = time.split(':')
    seconds = float(s[2]) + float(s[1]) * 60 + float(s[0]) * 3600
    return seconds
