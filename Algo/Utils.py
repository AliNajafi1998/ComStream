from math import sqrt, log
from DataAgent import DataAgent


def get_distance_tf_itf_cosine(data_agent: DataAgent, freq1: dict, freq2: dict):
    idf = calculate_itf(data_agent, freq1, freq2)

    sum_freq1 = sum(freq1.values())
    sum_freq2 = sum(freq2.values())

    sum_product_dots = 0.0
    dp1_tf_itf_length = 0.0
    dp2_tf_itf_length = 0.0
    for term_in_dp1 in freq1:
        if term_in_dp1 in idf:
            if term_in_dp1 in freq2:
                sum_product_dots += (freq1[term_in_dp1] / sum_freq1 * idf[term_in_dp1]) * (
                        freq2[term_in_dp1] / sum_freq2 * idf[term_in_dp1])
            dp1_tf_itf_length += pow(freq1[term_in_dp1] / sum_freq1 * idf[term_in_dp1], 2)

    dp1_tf_itf_length = sqrt(dp1_tf_itf_length)

    for term_in_dp2 in freq2:
        if term_in_dp2 in idf:
            dp2_tf_itf_length += pow(freq2[term_in_dp2] / sum_freq2 * idf[term_in_dp2], 2)
    dp2_tf_itf_length = sqrt(dp2_tf_itf_length)

    if dp1_tf_itf_length < data_agent.epsilon or dp2_tf_itf_length < data_agent.epsilon:
        return 1.0
    else:
        return 1.0 - (sum_product_dots / (dp1_tf_itf_length * dp2_tf_itf_length))


def calculate_itf(data_agent: DataAgent, freq1: dict, freq2: dict):
    terms_global_frequency = data_agent.terms_global_frequency
    itf_dictionary = dict()

    # Handling New DataPoint Frequencies
    for token_id, frequency in freq1.items():
        if token_id not in data_agent.global_freq:
            global_tf_freq = frequency
        else:
            global_tf_freq = data_agent.global_freq[token_id] + frequency
        terms_global_frequency += frequency
        if global_tf_freq < data_agent.epsilon:
            itf_dictionary[token_id] = 0.0
        else:
            itf_dictionary[token_id] = 1.0 + log(terms_global_frequency / global_tf_freq)

    # Handling old frequencies
    for token_id, frequency in freq2.items():
        if token_id not in itf_dictionary:
            if data_agent.global_freq[token_id] < data_agent.epsilon:
                itf_dictionary[token_id] = 0.0
            else:
                itf_dictionary[token_id] = 1.0 + log(terms_global_frequency / data_agent.global_freq[token_id])

    return itf_dictionary


def get_seconds(time: str):
    s = time.split(':')
    seconds = float(s[2]) + float(s[1]) * 60 + float(s[0]) * 3600
    return seconds
