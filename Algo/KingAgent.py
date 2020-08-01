import copy

from DataAgent import DataAgent
from Agent import Agent
from Utils import get_distance_tf_idf_cosine, get_seconds
import random
import re
import time
import pickle
import pandas as pd
import os
from math import log


class KingAgent:
    prev_residual = 0
    date = pd.to_datetime('2000-01-29T00:00:00Z')
    prev_data = None

    def __init__(self,
                 max_topic_count: int,
                 communication_step: str,
                 clean_up_step: str,
                 radius: float,
                 alpha: int,
                 outlier_threshold: float,
                 top_n: int,
                 dp_count: int,
                 fading_rate,
                 generic_distance=get_distance_tf_idf_cosine):

        pattern = re.compile(r'^[0-9]+:[0-9]{2}:[0-9]{2}$')
        are_invalid_steps = len(pattern.findall(communication_step)) != 1 or len(pattern.findall(clean_up_step)) != 1

        if are_invalid_steps:
            raise Exception(f'Invalid inputs fot steps')

        self.agents = {}
        self.radius = radius
        self.fading_rate = fading_rate
        self.communication_step = communication_step
        self.alpha = alpha
        self.max_topic_count = max_topic_count
        self.outlier_threshold = outlier_threshold
        self.top_n = top_n
        self.clean_up_deltatime = clean_up_step
        self.data_agent = DataAgent(count=dp_count)
        self.generic_distance_function = generic_distance
        self.dp_id_to_agent_id = dict()

    def create_agent(self) -> int:
        agent = Agent(self, generic_distance_function=self.generic_distance_function)
        self.agents[agent.agent_id] = agent
        return agent.agent_id

    def remove_agent(self, agent_id) -> None:
        for dp_id in self.agents[agent_id].dp_ids:
            self.agents[agent_id].remove_data_point(dp_id)
        del self.agents[agent_id]

    def handle_outliers(self) -> None:
        outliers_id = []
        for agent_id in copy.deepcopy(self.agents):
            outliers_id.extend(self.agents[agent_id].get_outliers())
            if len(self.agents[agent_id].dp_ids) < 1:
                self.remove_agent(agent_id)
        outliers_to_join = []
        for outlier_id in outliers_id:
            min_distance = float('infinity')
            similar_agent_id = -1
            for agent_id, agent in self.agents.items():
                distance = agent.get_distance(self, self.data_agent.data_points[outlier_id].freq)
                if distance <= min_distance:
                    min_distance = distance
                    similar_agent_id = agent_id
            if similar_agent_id != -1:
                outliers_to_join.append((outlier_id, min_distance, similar_agent_id))
            else:
                print('Sth went wrong!')
        outliers_to_join = sorted(outliers_to_join, key=lambda tup: tup[1])
        if self.top_n < len(outliers_to_join):
            outliers_to_join = outliers_to_join[:self.top_n]

        for dp_id, distance, agent_id in outliers_to_join:
            if distance > self.radius:
                new_agent_id = self.create_agent()
                self.agents[new_agent_id].add_data_point(self.data_agent.data_points[dp_id])
            else:
                self.agents[agent_id].add_data_point(self.data_agent.data_points[dp_id])

    def warm_up(self):
        for i in range(self.max_topic_count):
            self.create_agent()

        agents_dict = {id_: self.alpha for id_ in self.agents.keys()}
        for i in range(self.max_topic_count * self.alpha):
            # if KingAgent.prev_data != KingAgent.date:
            #     for dp_id, agent_id in self.agents.items():
            #         if os.path.isfile(os.path.join(os.getcwd(), 'dp_tracking.csv')):
            #             df = pd.read_csv(os.path.join(os.getcwd(), 'dp_tracking.csv'))
            #         else:
            #             df = pd.DataFrame(columns=['dp_id'])

            random_agent_id = random.sample(list(agents_dict), k=1)[0]
            dp = self.data_agent.get_next_dp()
            self.agents[random_agent_id].add_data_point(dp)
            agents_dict[random_agent_id] -= 1
            if agents_dict[random_agent_id] == 0:
                del agents_dict[random_agent_id]
        del agents_dict

    def stream(self):
        dp = self.data_agent.get_next_dp()
        min_distance = float('infinity')
        similar_agent_id = -1
        for agent_id, agent in self.agents.items():
            distance = agent.get_distance(self, self.data_agent.data_points[dp.dp_id].freq)
            if distance <= min_distance:
                min_distance = distance
                similar_agent_id = agent_id
        if min_distance > self.radius:
            new_agent_id = self.create_agent()
            self.agents[new_agent_id].add_data_point(self.data_agent.data_points[dp.dp_id])
        else:
            self.agents[similar_agent_id].add_data_point(self.data_agent.data_points[dp.dp_id])

    def fade_agents(self):
        for agent_id in list(self.agents.keys()):
            agent = self.agents[agent_id]
            agent.fade_agent(self.fading_rate)
            if agent.weight < self.data_agent.epsilon:
                self.remove_agent(agent_id)

    def handle_old_dps(self):
        for agent_id, agent in self.agents.items():
            agent.handle_old_dps()

    def train(self):
        self.warm_up()
        self.handle_outliers()

        while self.data_agent.has_next_dp():
            print(f'number of agents : {len(self.agents)}')
            self.stream()

            residual = time.mktime(KingAgent.date.timetuple()) % get_seconds(self.communication_step)
            if residual < KingAgent.prev_residual:
                KingAgent.prev_residual = 0
                self.handle_old_dps()
                self.handle_outliers()
                self.fade_agents()
            KingAgent.prev_residual = residual

    def save_model(self, parent_dir):
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        with open(os.path.join(parent_dir, 'model.pkl'), 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_model(cls, file_dir):
        with open(file_dir, 'rb') as file:
            return pickle.load(file)

    def write_output_to_files(self, parent_dir):
        for agent_id, agent in self.agents.items():
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            with open(os.path.join(parent_dir, f"{agent_id}.txt"), 'w') as file:
                for dp_id in agent.dp_ids:
                    dp_df = self.data_agent.raw_data.iloc[[self.data_agent.data_points[dp_id].index_in_df]]
                    file.write(str(dp_df['TEXT'].values[0]) + '\n')

    def get_topics_of_agents(self, max_topic_n=10):
        agent_topics = {}
        for agent_id, agent in self.agents.items():
            tf_idf = {}
            for term_id, f in agent.agent_global_f.items():
                dfi = 0
                for agent_id_2, agent_2 in self.agents.items():
                    if term_id in agent_2.agent_global_f:
                        dfi += 1

                tf_idf[term_id] = 1 + log((len(self.agents) + 1) / dfi) * (f / sum(agent.agent_global_f.values()))

            agent_topics[agent_id] = sorted(tf_idf.items(), key=lambda x: -x[1])[:max_topic_n]
        return agent_topics
