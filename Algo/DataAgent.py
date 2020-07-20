from .Singelton import Singleton
from .DataPoint import DataPoint
import pandas as pd
from datetime import datetime
from collections import defaultdict

from os import getcwd, path, chdir


@Singleton
class DataAgent:
    date = pd.to_datetime('2000-05-29T00:00:12Z')
    token_id = 0
    current_dp_index = 0
    terms_global_frequency = 0

    def __init__(self, count: int, epsilon=1e-3):
        self.epsilon = epsilon
        self.raw_data = None
        self.token_to_id = {}
        self.global_tf = {}
        self.count = count
        self.data_points = {}

        chdir('..')
        self.load_data(path.join(getcwd(), 'Data/data_cleaned1.pkl'), count=count)
        chdir('./Algo')

    def load_data(self, file_path: str, count: int) -> None:
        self.raw_data = pd.read_csv(file_path).head(count)

    def get_dp(self, dp: pd.DataFrame) -> DataPoint:
        tweet = dp['text'].values[0]

        # Extracting Data
        tf_dict = self.get_tf_dict(tweet)

        time_stamp = datetime.now()
        user_id = dp['user_id'].values[0]
        status_id = dp['status_id'].values[0]
        created_at = dp['created_at'].values[0]
        is_verified = dp['verified'].values[0]
        favourites_count = dp['favourites_count'].values[0]
        retweet_count = dp['retweet_count'].values[0]

        # Updating Current Date
        DataAgent.date = pd.to_datetime(created_at)

        return DataPoint(
            tf=tf_dict, time_stamp=time_stamp,
            user_id=user_id, status_id=status_id,
            created_at=created_at, is_verified=is_verified,
            favourites_count=favourites_count, retweet_count=retweet_count)

    def get_tf_dict(self, tweet: str) -> dict:
        tweet_tokens = tweet.split()

        tf_dict = defaultdict(0)
        for token in tweet_tokens:
            if token in self.token_to_id:
                tf_dict[token] += 1
            else:
                self.token_to_id[token] = DataAgent.token_id
                DataAgent.token_id += 1
        return tf_dict

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
