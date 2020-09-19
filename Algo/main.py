from KingAgent import KingAgent
from Utils import get_distance_tf_idf_cosine
import os

if __name__ == '__main__':
    os.chdir('..')
    data_path = os.path.join(os.getcwd(), 'Data/FA/FACUP.pkl')
    king = KingAgent(max_topic_count=5,
                     save_output_interval="00:01:00",
                     communication_step="00:01:00",
                     clean_up_step="00:02:00",
                     radius=0.70,  # 75 for reuters
                     alpha=2,
                     outlier_threshold=0.73,
                     top_n=100,
                     dp_count=1000000,
                     max_keyword_per_agent=5,
                     fading_rate=0.0,  # fading_rate amount gets faded # 0.5
                     delete_faded_threshold=0.0, # 0.4
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
