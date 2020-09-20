from Algo.KingAgent import KingAgent
from Algo.Utils import get_distance_tf_idf_cosine
import os

if __name__ == '__main__':
    os.chdir('..')
    data_path = os.path.join(os.getcwd(), 'Data/data_cleaned.pkl')
    king = KingAgent(init_no_agents=5,
                     save_output_interval="24:00:00",
                     communication_interval="01:30:00",
                     sliding_window_interval="24:00:00",
                     radius=0.75,
                     outlier_threshold=0.78,
                     init_dp_per_agent=2,
                     dp_count=10000000,
                     max_no_keywords=5,
                     agent_fading_rate=0.5,  # agent_fading_rate amount gets faded # 0.5
                     delete_agent_weight_threshold=0.4,  # 0.4
                     generic_distance=get_distance_tf_idf_cosine,
                     is_twitter=True,
                     data_file_path=data_path,
                     verbose=1
                     )
    os.chdir('./Algo')
    king.train()
    # king = KingAgent.load_model(os.getcwd() + '/model/model.pkl')

    # king.save_model(os.path.join(os.getcwd(), 'model'))
    # king.write_output_to_files(os.path.join(os.getcwd(), 'last_output/output'))
    # king.write_topics_to_files(os.path.join(os.getcwd(), 'last_output/topics'), 5)
