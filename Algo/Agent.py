from typing import Callable

from DataPoint import DataPoint
from Utils import get_seconds


class Agent:
    agent_id = 0

    def __init__(self, king_agent, generic_distance_function: Callable):
        self.agent_id = Agent.agent_id
        self.outlier_threshold = king_agent.outlier_threshold
        Agent.agent_id += 1
        self.agent_global_f = {}
        self.weight = 1
        self.dp_ids = []
        self.king_agent = king_agent
        self.generic_distance_function = generic_distance_function

    def add_data_point(self, dp: DataPoint) -> None:
        """
        Adding data point to the agent
        :param dp: data point We want to add to the Agent
        :return: None
        """
        self.weight += 1
        for token_id, frequency in dp.freq.items():
            if token_id in self.agent_global_f:
                self.agent_global_f[token_id] += frequency
                self.update_global_tf(frequency, token_id)
            else:
                self.agent_global_f[token_id] = frequency
                self.update_global_tf(frequency, token_id)

        self.dp_ids.append(dp.dp_id)

    def update_global_tf(self, frequency, token_id):
        if token_id in self.king_agent.data_agent.global_freq:
            self.king_agent.data_agent.global_freq[token_id] += frequency
            self.king_agent.data_agent.terms_global_frequency += frequency
        else:
            self.king_agent.data_agent.global_freq[token_id] = frequency
            self.king_agent.data_agent.terms_global_frequency += frequency

    def remove_data_point(self, dp_id: int) -> None:
        """
        Removing data point from agent
        :param dp_id: Data Point id
        :return: None
        """
        try:
            self.dp_ids.remove(dp_id)
            self.weight -= 1
            if self.weight <= 0:
                self.weight = 0
            # print(self.king_agent.data_agent.data_points.keys())
            for token_id, frequency in self.king_agent.data_agent.data_points[dp_id].freq.items():
                self.king_agent.data_agent.global_freq[token_id] -= frequency
                self.king_agent.data_agent.terms_global_frequency -= frequency
            del self.king_agent.data_agent.data_points[dp_id]
        except ValueError:
            print(f'There is no such data point in Agent : {dp_id}')

    def fade_agent(self, fade_rate: float) -> None:
        """
        Fading Agent Weight
        :param fade_rate: float number between 0 and 1
        :return: None
        """
        if fade_rate > 1 or fade_rate < 0:
            raise Exception(f'Invalid Fade Rate : {fade_rate}')
        else:
            self.weight = self.weight * (1 - fade_rate)

    def get_outliers(self) -> list:
        """
        Getting outliers of agent
        :return: list of ids of outliers
        """
        outliers_id = []
        for dp_id in self.dp_ids:
            dp = self.king_agent.data_agent.data_points[dp_id]
            distance = self.get_distance(self.king_agent, dp.freq)
            if distance > self.outlier_threshold:
                self.dp_ids.remove(dp_id)
                outliers_id.append(dp_id)
        return outliers_id

    def get_distance(self, king_agent, f: dict):
        return self.generic_distance_function(king_agent, f, self.agent_global_f)

    def handle_old_dps(self):
        for dp_id in self.dp_ids:
            dp = self.king_agent.data_agent.data_points[dp_id]
            if abs((dp.created_at - self.king_agent.data_agent.date).total_seconds()) > get_seconds(
                    self.king_agent.clean_up_deltatime):
                self.remove_data_point(dp_id)
