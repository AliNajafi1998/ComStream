from KingAgent import KingAgent
from Utils import get_distance_tf_idf_cosine
import os

if __name__ == '__main__':
    os.chdir('..')
    data_path = os.path.join(os.getcwd(), 'Data/data_cleaned.pkl')
    king = KingAgent(max_topic_count=5,
                     communication_step="00:00:10",
                     clean_up_step="00:20:10",
                     radius=0.7,
                     alpha=5,
                     outlier_threshold=0.8,
                     top_n=30,
                     dp_count=200,
                     fading_rate=0,
                     generic_distance=get_distance_tf_idf_cosine,
                     is_twitter=True,
                     data_file_path=data_path
                     )
    os.chdir('./Algo')
    king.train()
    # king = KingAgent.load_model(os.getcwd() + '/model/model.pkl')

    king.save_model(os.path.join(os.getcwd(), 'model'))
    king.write_output_to_files(os.path.join(os.getcwd(), 'output'))
    king.write_topics_to_files(os.path.join(os.getcwd(), 'topics'), 5)
