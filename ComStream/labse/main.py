from .Coordinator import Coordinator
from .Utils import get_distance_cosine
import os

if __name__ == '__main__':
    data_path = os.path.join('/ComStream/ComStream/Data/data_cleaned.pkl')  # The file must be pickle file :)
    embedding_file_path = os.path.join("embeds.npy")
    coordinator = Coordinator(
        init_no_agents=5,
        init_dp_per_agent=2,
        save_output_interval="00:01:00",
        communication_interval="00:01:00",
        sliding_window_interval="00:01:00",
        assign_radius=0.80,
        outlier_threshold=0.83,
        no_topics=10,
        no_keywords=5,
        agent_fading_rate=0.5,
        delete_agent_weight_threshold=0.4,
        generic_distance=get_distance_cosine,
        data_file_path=data_path,
        embedding_file_path=embedding_file_path,
        dp_count=10000000,
        verbose=1
    )
    coordinator.train()
