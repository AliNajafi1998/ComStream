from .Singelton import Singleton
from .DataPoint import DataPoint
import pandas as pd
from datetime import datetime


@Singleton
class DataAgent:
    token_id = 0
    current_dp_index = 0

    def __init__(self):
        self.raw_data = None
        self.token_to_id = {}
        self.data_points = []

    def load_data(self, path: str) -> None:
        self.raw_data = pd.read_csv(path)

    def vectorize_data(self) -> None:
        for i in range(len(self.raw_data)):
            item = self.raw_data.iloc[[i]]
            tweet = item['text'].values[0]

            # Extracting Data
            tf_dict = self.get_tf_dict(tweet)
            time_stamp = datetime.now()
            user_id = item['user_id'].values[0]
            status_id = item['status_id'].values[0]
            created_at = item['created_at'].values[0]
            is_verified = item['verified'].values[0]
            favourites_count = item['favourites_count'].values[0]
            retweet_count = item['retweet_count'].values[0]

            # Creating DataPoint
            dp = DataPoint(
                tf=tf_dict, time_stamp=time_stamp,
                user_id=user_id, status_id=status_id,
                created_at=created_at, is_verified=is_verified,
                favourites_count=favourites_count, retweet_count=retweet_count
            )
            self.data_points.append(dp)

    def get_tf_dict(self, tweet: str) -> dict:
        tweet_tokens = tweet.split()
        tf_dict = {}
        for token in tweet_tokens:
            if token in self.token_to_id:
                if token in tf_dict:
                    tf_dict[token] += 1
                else:
                    tf_dict[token] = 1
            else:
                self.token_to_id[token] = DataAgent.token_id
                DataAgent.token_id += 1
        return tf_dict

    def get_next_dp(self):
        if DataAgent.current_dp_index >= len(self.data_points):
            print('Finished')
            return None
        else:
            DataAgent.current_dp_index += 1
            return self.data_points[DataAgent.current_dp_index - 1]
