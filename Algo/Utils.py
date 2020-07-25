from math import sqrt, log
from DataAgent import DataAgent


def get_distance_tf_itf_cosine(data_agent: DataAgent, tf1: dict, tf2: dict):
    idf = calculate_itf(data_agent, tf1, tf2)

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


def calculate_itf(data_agent: DataAgent, tf1: dict, tf2: dict):
    terms_global_frequency = data_agent.terms_global_frequency
    itf_dictionary = dict()

    for token_id, frequency in tf1.items():
        global_tf_freq = None
        if token_id not in data_agent.global_tf:
            global_tf_freq = frequency
        else:
            global_tf_freq = data_agent.global_tf[token_id]
        if global_tf_freq < data_agent.epsilon:
            itf_dictionary[token_id] = 0.0
        else:
            itf_dictionary[token_id] = 1.0 + log(terms_global_frequency / global_tf_freq)

    for token_id, frequency in tf2.items():
        if token_id not in itf_dictionary:
            if data_agent.global_tf[token_id] < data_agent.epsilon:
                itf_dictionary[token_id] = 0.0
            else:
                itf_dictionary[token_id] = 1.0 + log(terms_global_frequency / data_agent.global_tf[token_id])

    return itf_dictionary


def get_seconds(time: str):
    s = time.split(':')
    seconds = float(s[2]) + float(s[1]) * 60 + float(s[0]) * 3600
    return seconds
