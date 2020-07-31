class DataPoint:
    dp_id = 0

    def __init__(self,
                 freq: dict,
                 time_stamp,
                 topics,
                 created_at,
                 index_in_df):
        self.dp_id = DataPoint.dp_id
        DataPoint.dp_id += 1
        self.freq = freq
        self.created_at = created_at
        self.time_stamp = time_stamp
        self.topics = topics
        self.index_in_df = index_in_df
