from ComStream.labse.Coordinator import Coordinator
from ComStream.labse.Utils import get_distance_cosine
import os
from pathlib import Path
import time

if __name__ == '__main__':
    start_time = time.time()
    p = Path(__file__).parents[2]
    os.chdir(p)
    print(os.getcwd())
    data_path = os.path.join('Data/FA_Preprocessed.pkl')  # The file must be pickle file :)

    embedding_file_path = os.path.join("Data/FA_Embedding.npy")
    coordinator = Coordinator(
        init_no_agents=2,
        init_dp_per_agent=2,
        save_output_interval="00:01:00",
        communication_interval="00:01:00",
        sliding_window_interval="00:01:00",
        assign_radius=0.24,  # 0<x<1 | lower -> more clusters
        outlier_threshold=0.26,  # this should be a bit more than assign_radius
        no_topics=30,
        no_keywords=30,
        agent_fading_rate=0.0,
        delete_agent_weight_threshold=0.0,
        generic_distance=get_distance_cosine,
        data_file_path=data_path,
        embedding_file_path=embedding_file_path,
        dp_count=10000000,
        verbose=1,
        is_parallel=False
    )
    coordinator.train()
    end_time = time.time()
    print(f'Ran in {end_time - start_time} seconds')
# 20-22:
# 25-27:
# 30-32:
