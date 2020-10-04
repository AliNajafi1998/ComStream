from Algo.KingAgent import KingAgent
from Algo.Utils import get_distance_tf_idf_cosine
import os
from pathlib import Path

if __name__ == '__main__':
    # 'Data/data_cleaned_1000k.pkl'
    os.chdir("../Data")
    data_path = os.path.join(os.getcwd(), 'FA/FACUP.pkl')
    king = KingAgent(init_no_agents=5,
                     init_dp_per_agent=2,
                     save_output_interval="00:01:00",
                     communication_interval="00:01:00",
                     sliding_window_interval="00:01:00",
                     radius=0.80,
                     outlier_threshold=0.83,
                     max_no_topics=10,
                     max_no_keywords=5,
                     agent_fading_rate=0.5,
                     delete_agent_weight_threshold=0.4,
                     generic_distance=get_distance_tf_idf_cosine,
                     data_file_path=data_path,
                     dp_count=10000000,
                     verbose=1
                     )
    king.train()
