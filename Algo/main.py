from KingAgent import KingAgent
from Utils import get_distance_tf_idf_cosine

king = KingAgent(max_topic_count=10,
                 communication_step="00:00:10",
                 clean_up_step="00:20:10",
                 radius=0.5, alpha=2,
                 outlier_threshold=0.8,
                 top_n=10,
                 dp_count=100,
                 fading_rate=0.3,
                 generic_distance=get_distance_tf_idf_cosine)

king.train()

print(len(king.agents))
print(list(king.agents.values())[1].weight)
