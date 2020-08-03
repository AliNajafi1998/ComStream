class DataPoint:
    dp_id = 0

    def __init__(self,
                 freq: dict,
                 time_stamp,
                 created_at,
                 index_in_df):
        self.freq = freq
        self.created_at = created_at
        self.time_stamp = time_stamp
        self.index_in_df = index_in_df


class ReutersDataPoint(DataPoint):
    def __init__(self, freq: dict, time_stamp, topics, created_at, index_in_df):
        super().__init__(freq, time_stamp, created_at, index_in_df)
        self.dp_id = DataPoint.dp_id
        DataPoint.dp_id += 1
        self.topics = topics


class TwitterDataPoint(DataPoint):
    def __init__(self, freq: dict, time_stamp, user_id, status_id, created_at, is_verified, favourites_count,
                 retweet_count, index_in_df):
        super().__init__(freq, time_stamp, created_at, index_in_df)
        self.dp_id = DataPoint.dp_id
        DataPoint.dp_id += 1
        self.user_id = user_id
        self.status_id = status_id
        self.is_verified = is_verified
        self.favourites_count = favourites_count
        self.retweet_count = retweet_count
