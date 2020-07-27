from math import sqrt, log


def get_distance_tf_idf_cosine(king_agent, freq1: dict, freq2: dict):
    idf = calculate_idf(king_agent, freq1, freq2)

    sum_freq1 = sum(freq1.values())
    sum_freq2 = sum(freq2.values())

    sum_product_dots = 0.0
    dp1_tf_itf_length = 0.0
    dp2_tf_itf_length = 0.0
    for token_id_in_dp1 in freq1:
        if token_id_in_dp1 in idf:
            if token_id_in_dp1 in freq2:
                sum_product_dots += (freq1[token_id_in_dp1] / sum_freq1 * idf[token_id_in_dp1]) * (
                            freq2[token_id_in_dp1] / sum_freq2 * idf[token_id_in_dp1])
            dp1_tf_itf_length += pow(freq1[token_id_in_dp1] / sum_freq1 * idf[token_id_in_dp1], 2)

    dp1_tf_itf_length = sqrt(dp1_tf_itf_length)

    for token_id_in_dp2 in freq2:
        if token_id_in_dp2 in idf:
            dp2_tf_itf_length += pow(freq2[token_id_in_dp2] / sum_freq2 * idf[token_id_in_dp2], 2)
    dp2_tf_itf_length = sqrt(dp2_tf_itf_length)

    if dp1_tf_itf_length < king_agent.data_agent.epsilon or dp2_tf_itf_length < king_agent.data_agent.epsilon:
        return 1.0
    else:
        return 1.0 - (sum_product_dots / (dp1_tf_itf_length * dp2_tf_itf_length))


def calculate_idf(king_agent, freq1: dict, freq2: dict):
    idf_dictionary = dict()

    for token_id_1, frequency_1 in freq1.items():
        for token_id_2, frequency_2 in freq2.items():
            if token_id_1 == token_id_2:
                counter = 1
                for agent_id, agent in king_agent.agents.items():
                    if token_id_1 in agent.agent_global_f:
                        counter += 1
                idf_dictionary[token_id_1] = log(len(king_agent.agents) / counter)

    return idf_dictionary


def get_seconds(time: str):
    s = time.split(':')
    seconds = float(s[2]) + float(s[1]) * 60 + float(s[0]) * 3600
    return seconds
