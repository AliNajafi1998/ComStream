from .DataPoint import TwitterDataPoint

import pandas as pd
from datetime import datetime


class DataManager:
    token_id = 0
    current_dp_index = 0
    terms_global_frequency = 0

    def __init__(self, data_file_path, count: int, epsilon=1e-7):
        """
        the object that handles data loading and processing
        :param data_file_path: the path where our data is at
        :param count: the amount of the data we want to process
        :param epsilon: Float
        :return: None
        """
        self.data_file_path = data_file_path
        self.epsilon = epsilon
        self.raw_data = None
        self.token_to_id = {}
        self.id_to_token = {}
        self.global_freq = {}
        self.count = count
        self.data_points = {}

        self.load_data(data_file_path, count=count)

    def load_data(self, file_path: str, count: int) -> None:
        """
        read the df of our dps
        :param file_path: the dir where our input data is at
        :param count: how many of the data we want
        :return: None
        """
        self.raw_data = pd.read_pickle(file_path).reset_index().head(count)

    def get_dp(self, dp: pd.DataFrame):
        """
        call the function to turn the df to our dp object
        :param dp: one dp with format data frame
        :return: object dp
        """
        return self.get_twitter_dp(dp)

    def get_twitter_dp(self, dp: pd.DataFrame) -> TwitterDataPoint:
        """
        turns one twitter data point's data-frame to our recognizable object dp
        :param dp: one twitter dp with format data frame
        :return: object twitter dp
        """
        tweet = dp['text'].values[0]

        # Extracting Data
        freq_dict = self.get_freq_dict(tweet)

        time_stamp = datetime.now()
        status_id = dp['status_id'].values[0]
        created_at = pd.to_datetime(dp['created_at'].values[0])

        return TwitterDataPoint(
            tweet=tweet,
            freq=freq_dict, time_stamp=time_stamp,
            status_id=status_id,
            created_at=created_at,
            index_in_df=DataManager.current_dp_index - 1
        )

    def get_freq_dict(self, tweet: str) -> dict:
        """
        turns the tweet text in to its token ids with their frequencies
        :param tweet: the text of the tweet
        :return: a dict of {token_id:frequency}
        """
        tweet_tokens = tweet.split()

        freq_dict = {}
        for token in tweet_tokens:
            if token in self.token_to_id:
                if self.token_to_id[token] in freq_dict:
                    freq_dict[self.token_to_id[token]] += 1
                else:
                    freq_dict[self.token_to_id[token]] = 1
            else:
                self.token_to_id[token] = DataManager.token_id
                self.id_to_token[DataManager.token_id] = token
                DataManager.token_id += 1
        return freq_dict

    def get_next_dp(self):
        """
        call the func to read the next dp
        :return: the object of the dp
        """
        DataManager.current_dp_index += 1
        dp = self.get_dp(self.raw_data.iloc[[DataManager.current_dp_index - 1]])
        self.data_points[dp.dp_id] = dp
        return dp

    def has_next_dp(self):
        """
        check if we are not exceeding our maximum dps to process threshold
        :return: Boolean
        """
        return not (DataManager.current_dp_index >= self.count)
