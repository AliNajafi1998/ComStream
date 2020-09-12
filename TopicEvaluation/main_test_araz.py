import os
from pathlib import Path
from topic_extraction_corona import TopicExtractor
in_days_dir = os.path.join(Path(os.getcwd()).parent, 'Data/outputs/multi_agent')
out_save_dir = os.path.join(Path(os.getcwd()).parent, 'Data/outputs/multi_agent_topics')
TE = TopicExtractor(data_dir=in_days_dir, save_dir=out_save_dir, top_n_topics=10, top_n_keywords=10)
TE.extract_and_save_in_files()
