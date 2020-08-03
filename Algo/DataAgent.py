import copy

from DataPoint import ReutersDataPoint
from DataPoint import TwitterDataPoint

import pandas as pd
from datetime import datetime


class DataAgent:
    token_id = 0
    current_dp_index = 0
    terms_global_frequency = 0

    def __init__(self, data_file_path, count: int, epsilon=1e-7, is_twitter=True):
        self.data_file_path = data_file_path
        self.is_twitter = is_twitter
        self.epsilon = epsilon
        self.raw_data = None
        self.token_to_id = {}
        self.id_to_token = {}
        self.global_freq = {}
        self.count = count
        self.data_points = {}

        self.load_data(data_file_path, count=count)

    def load_data(self, file_path: str, count: int) -> None:
        self.raw_data = pd.read_pickle(file_path).reset_index().head(count)

    def get_dp(self, dp: pd.DataFrame):
        if self.is_twitter:
            return self.get_twitter_dp(dp)
        else:
            return self.get_reuters_dp(dp)

    def get_twitter_dp(self, dp: pd.DataFrame) -> TwitterDataPoint:
        tweet = dp['text'].values[0]

        # Extracting Data
        freq_dict = self.get_freq_dict(tweet)

        time_stamp = datetime.now()
        user_id = dp['user_id'].values[0]
        status_id = dp['status_id'].values[0]
        created_at = pd.to_datetime(dp['created_at'].values[0])
        is_verified = dp['verified'].values[0]
        favourites_count = dp['favourites_count'].values[0]
        retweet_count = dp['retweet_count'].values[0]

        # Updating Current Date
        from KingAgent import KingAgent
        KingAgent.prev_data = copy.deepcopy(KingAgent.date)
        KingAgent.date = pd.to_datetime(created_at)

        return TwitterDataPoint(
            freq=freq_dict, time_stamp=time_stamp,
            user_id=user_id, status_id=status_id,
            created_at=created_at, is_verified=is_verified,
            favourites_count=favourites_count, retweet_count=retweet_count,
            index_in_df=DataAgent.current_dp_index - 1
        )

    def get_reuters_dp(self, dp: pd.DataFrame) -> ReutersDataPoint:
        tweet = dp['TEXT'].values[0]

        # Extracting Data
        freq_dict = self.get_freq_dict(tweet)

        time_stamp = datetime.now()
        topics = dp['TOPICS']
        created_at = pd.to_datetime(dp['CREATED_AT'].values[0])

        # Updating Current Date
        from KingAgent import KingAgent
        KingAgent.prev_data = copy.deepcopy(KingAgent.date)
        KingAgent.date = created_at

        return ReutersDataPoint(
            freq=freq_dict,
            time_stamp=time_stamp,
            topics=topics,
            created_at=created_at,
            index_in_df=DataAgent.current_dp_index - 1
        )

    def get_freq_dict(self, tweet: str) -> dict:
        tweet_tokens = tweet.split()

        freq_dict = {}
        for token in tweet_tokens:
            if token in self.token_to_id:
                if self.token_to_id[token] in freq_dict:
                    freq_dict[self.token_to_id[token]] += 1
                else:
                    freq_dict[self.token_to_id[token]] = 1
            else:
                self.token_to_id[token] = DataAgent.token_id
                self.id_to_token[DataAgent.token_id] = token
                DataAgent.token_id += 1
        return freq_dict

    def get_next_dp(self):
        if DataAgent.current_dp_index >= self.count:
            print('Finished')
            return None
        else:
            DataAgent.current_dp_index += 1
            dp = self.get_dp(self.raw_data.iloc[[DataAgent.current_dp_index - 1]])
            self.data_points[dp.dp_id] = dp
            return dp

    def has_next_dp(self):
        if DataAgent.current_dp_index >= self.count:
            return False
        else:
            return True
