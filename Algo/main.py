from Algo.KingAgent import KingAgent
from Algo.Utils import get_distance_tf_idf_cosine
import os

if __name__ == '__main__':
    os.chdir('..')
    data_path = os.path.join(os.getcwd(), 'Data/data_cleaned_1000k.pkl')
    king = KingAgent(init_no_agents=5,
                     init_dp_per_agent=2,
                     save_output_interval="00:30:00",
                     communication_interval="00:10:00",
                     sliding_window_interval="24:00:00",
                     radius=0.75,
                     outlier_threshold=0.78,
                     max_no_topics=10,
                     max_no_keywords=5,
                     agent_fading_rate=0.5,  # agent_fading_rate amount gets faded # 0.5
                     delete_agent_weight_threshold=0.4,  # 0.4
                     generic_distance=get_distance_tf_idf_cosine,
                     is_twitter=True,
                     data_file_path=data_path,
                     dp_count=10000000,
                     verbose=1
                     )
    os.chdir('./Algo')
    king.train()

