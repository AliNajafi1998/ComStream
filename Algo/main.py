from KingAgent import KingAgent
from Utils import get_distance_tf_idf_cosine
import os

if __name__ == '__main__':
    os.chdir('..')
    data_path = os.path.join(os.getcwd(), 'Data/data_cleaned.pkl')
    king = KingAgent(max_topic_count=5,
                     save_output_interval="00:02:00",
                     communication_step="00:01:00",
                     clean_up_step="24:00:00",
                     radius=0.75,  # 75 for reuters
                     alpha=2,
                     outlier_threshold=0.80,
                     top_n=4,
                     dp_count=700000,
                     # fading_rate=1-0.2,  # it gets: w*= 1-fading_rate, 0.9 remains if it is 0.1
                     # delete_faded_threshold=0.70,
                     fading_rate=1 - 0.33,  # it gets: w*= 1-fading_rate, 0.9 remains if it is 0.1
                     delete_faded_threshold=0.50,
                     # fading_rate=0.0,  # it gets: w*= 1-fading_rate, 0.66 remains if it is 0.33
                     # delete_faded_threshold=0.0,
                     generic_distance=get_distance_tf_idf_cosine,
                     is_twitter=True,
                     data_file_path=data_path
                     )
    os.chdir('./Algo')
    king.train()
    # king = KingAgent.load_model(os.getcwd() + '/model/model.pkl')

    # king.save_model(os.path.join(os.getcwd(), 'model'))
    # king.write_output_to_files(os.path.join(os.getcwd(), 'last_output/output'))
    # king.write_topics_to_files(os.path.join(os.getcwd(), 'last_output/topics'), 5)
