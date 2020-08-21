import copy
from math import sqrt, log


def get_distance_tf_idf_cosine(king_agent, freq1: dict, freq2: dict):
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
    idf_dictionary = dict()
    freq = copy.deepcopy(freq1)
    for token_id, f1 in freq2.items():
        freq[token_id] = freq.get(token_id, 0) + freq2[token_id]

    for token_id_1, frequency_1 in freq.items():
        counter = 1 + king_agent.global_idf_count.get(token_id_1, 0)
        num = 1 + len(king_agent.agents)
        idf_dictionary[token_id_1] = 1 + log(num / counter)

    return idf_dictionary


def get_seconds(time: str):
    s = time.split(':')
    seconds = float(s[2]) + float(s[1]) * 60 + float(s[0]) * 3600
    return seconds
