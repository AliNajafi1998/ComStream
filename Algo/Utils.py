from math import sqrt, log
from .DataPoint import DataPoint
from .DataAgent import DataAgent


def get_distance_tf_idf_cosine(data_agent: DataAgent, dp1: DataPoint, dp2: DataPoint):
    idf = calculate_idf(data_agent, dp1, dp2)

    sum_product_dots = 0.0
    dp1_tf_idf_length = 0.0
    dp2_tf_idf_length = 0.0
    for term_in_dp1 in dp1.tf:
        if term_in_dp1 in idf:
            if term_in_dp1 in dp2.tf:
                sum_product_dots += (dp1.tf[term_in_dp1] * idf[term_in_dp1]) * (dp2.tf[term_in_dp1] * idf[term_in_dp1])
            dp1_tf_idf_length += pow(dp1.tf[term_in_dp1] * idf[term_in_dp1], 2)

    dp1_tf_idf_length = sqrt(dp1_tf_idf_length)

    for term_in_dp2 in dp2.tf:
        if term_in_dp2 in idf:
            dp2_tf_idf_length += pow(dp2.tf[term_in_dp2] * idf[term_in_dp2], 2)
    dp2_tf_idf_length = sqrt(dp2_tf_idf_length)

    if dp1_tf_idf_length < data_agent.epsilon or dp2_tf_idf_length < data_agent.epsilon:
        return 1.0
    else:
        return 1.0 - (sum_product_dots / (dp1_tf_idf_length * dp2_tf_idf_length))


def calculate_idf(data_agent: DataAgent, dp1, dp2):
    terms_global_frequency = data_agent.terms_global_frequency
    idf_dictionary = dict()

    for term in dp1.tf:
        if data_agent.global_tf[term] < data_agent.epsilon:
            idf_dictionary[term] = 0.0
        else:
            idf_dictionary[term] = 1.0 + log(terms_global_frequency / data_agent.global_tf[term])

    for term in dp2.tf:
        if term not in idf_dictionary:
            if data_agent.global_tf[term] < data_agent.epsilon:
                idf_dictionary[term] = 0.0
            else:
                idf_dictionary[term] = 1.0 + log(terms_global_frequency / data_agent.global_tf[term])

    return idf_dictionary
