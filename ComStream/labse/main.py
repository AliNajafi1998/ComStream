from ComStream.ComStream.labse.Coordinator import Coordinator
from ComStream.ComStream.labse.Utils import get_distance_cosine
import os
from pathlib import Path

if __name__ == '__main__':
    p = Path(__file__).parents[2]
    os.chdir(p)
    print(os.getcwd())
    data_path = os.path.join('Data/data_cleaned.pkl')  # The file must be pickle file :)

    embedding_file_path = os.path.join("Data/embeds.npy")
    coordinator = Coordinator(
        init_no_agents=2,
        init_dp_per_agent=2,
        save_output_interval="00:01:00",
        communication_interval="00:01:00",
        sliding_window_interval="00:01:00",
        assign_radius=0.80,
        outlier_threshold=0.83,
        no_topics=2,
        no_keywords=2,
        agent_fading_rate=0.5,
        delete_agent_weight_threshold=0.4,
        generic_distance=get_distance_cosine,
        data_file_path=data_path,
        embedding_file_path=embedding_file_path,
        dp_count=10000000,
        verbose=1
    )
    coordinator.train()
