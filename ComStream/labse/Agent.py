from typing import Callable
import numpy as np

from .Utils import get_seconds


class Agent:
    agent_id = 0
    epsilon = 1e-7

    def __init__(self, coordinator, generic_distance_function: Callable):
        self.agent_id = Agent.agent_id
        self.outlier_threshold = coordinator.outlier_threshold
        Agent.agent_id += 1
        self.centroid = np.zeros(768)
        self.weight = 0
        self.dp_ids = []
        self.coordinator = coordinator
        self.generic_distance_function = generic_distance_function

    def add_data_point(self, dp) -> None:
        """
        adding dp to the agent
        :param dp: dp we want to add to the agent
        :return: None
        """
        self.weight += 1
        self.centroid = (self.centroid + dp.embedding_vec) / 2
        self.dp_ids.append(dp.dp_id)
        self.coordinator.dp_id_to_agent_id[dp.dp_id] = self.agent_id

    def remove_data_point(self, dp_id: int, outlier=False) -> None:
        """
        removing data point from agent
        :param dp_id: dp id
        :param outlier : Boolean
        :return: None
        """
        try:
            self.centroid = (self.centroid * len(self.dp_ids) - self.coordinator.data_agent.data_points[dp_id]) / len(
                self.dp_ids)
            self.dp_ids.remove(dp_id)
            if self.weight <= 0:
                self.weight = 0
            if not outlier:
                del self.coordinator.data_agent.data_points[dp_id]
            del self.coordinator.dp_id_to_agent_id[dp_id]

        except ValueError:
            print(f'There is no such data point in Agent : {dp_id}')

    def get_outliers(self, out) -> None:
        """
        getting outliers of agent
        :return: list of ids of outliers
        """
        outliers_id = []
        for dp_id in self.dp_ids:
            dp = self.coordinator.data_agent.data_points[dp_id]
            distance = self.get_distance(self.coordinator, dp.embedding_vec)
            if distance > self.outlier_threshold:
                self.remove_data_point(dp_id, outlier=True)
                outliers_id.append(dp_id)
        out.extend(outliers_id)

    def get_distance(self, coordinator, embedding: np.array):
        """
        calls the function that finds the distance
        :param coordinator: the object of KingAgent
        :param embedding: embedding vector of datapoint
        :return: (float) returns the distance of the dp and this agent
        """
        return self.generic_distance_function(coordinator, embedding, self.centroid)

    def handle_old_dps(self):
        """
        deletes the dps that are older than sliding_window_interval time interval
        :return: None
        """
        for dp_id in self.dp_ids:
            dp = self.coordinator.data_agent.data_points[dp_id]
            if abs((dp.created_at - self.coordinator.current_date).total_seconds()) > get_seconds(
                    self.coordinator.sliding_window_interval):
                self.remove_data_point(dp_id)

    def fade_agent_weight(self, fade_rate: float, delete_faded_threshold: float) -> None:
        """
        fade an agent's weight
        :param fade_rate: the amount to be faded
        :param delete_faded_threshold: delete the agent if it's weight gets less than this threshold
        :return: None
        """
        if abs(fade_rate) < 1e-9:
            pass
        else:
            if fade_rate > 1 or fade_rate < 0 or delete_faded_threshold > 1 or delete_faded_threshold < 0:
                message = f'Invalid Fade Rate or delete_agent_weight_threshold : {fade_rate, delete_faded_threshold}'
                raise Exception(message)
            else:
                self.weight = self.weight * (1 - fade_rate)
                if self.weight < delete_faded_threshold:
                    self.coordinator.remove_agent(self.agent_id)
