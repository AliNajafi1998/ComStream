from .DataPoint import DataPoint


class Agent:
    agent_id = 0

    def __init__(self):
        self.agent_id = Agent.agent_id
        Agent.agent_id += 1
        self.global_tf = {}
        self.weight = 1
        self.dp_ids = []

    def add_data_point(self, dp: DataPoint) -> None:
        """
        Adding data point to the agent
        :param dp: data point We want to add to the Agent
        :return: None
        """
        self.weight += 1
        for token_id, frequency in dp.tf.items():
            if token_id in self.global_tf:
                self.global_tf[token_id] += frequency
            else:
                self.global_tf[token_id] = frequency
        self.dp_ids.append(dp.dp_id)

    def remove_data_point(self, dp_id: int) -> None:
        """
        Removing data point from agent
        :param dp_id: Data Point id
        :return: None
        """
        try:
            self.dp_ids.remove(dp_id)
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

