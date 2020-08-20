from Agent import Agent
from Utils import get_distance_tf_idf_cosine, get_seconds
import random
import re
import time
import pickle
import pandas as pd
import os
from math import log
import heapq
from threading import Thread
from DataAgent import DataAgent
from pathlib import Path
import copy


class KingAgent:
    current_date = pd.to_datetime('2020-03-29T00:00:00Z')

    prev_residual = 0
    save_output_prev_residual = 0
    date = pd.to_datetime('2000-01-29T00:00:00Z')
    dp_now = 0
    full_date = '2000-01-29T00:00:00Z'
    prev_full_date = '2000-01-29T00:00:00Z'

    def __init__(self,
                 save_output_interval: str,
                 max_topic_count: int,
                 communication_step: str,
                 clean_up_step: str,
                 radius: float,
                 alpha: int,
                 outlier_threshold: float,
                 top_n: int,
                 dp_count: int,
                 fading_rate: float,
                 delete_faded_threshold: float,
                 data_file_path: str,
                 is_twitter=False,
                 generic_distance=get_distance_tf_idf_cosine):

        pattern = re.compile(r'^[0-9]+:[0-9]{2}:[0-9]{2}$')
        are_invalid_steps = len(pattern.findall(communication_step)) != 1 or len(pattern.findall(clean_up_step)) != 1

        if are_invalid_steps:
            raise Exception(f'Invalid inputs fot steps')
        self.save_output_interval = save_output_interval
        self.is_twitter = is_twitter
        self.agents = {}
        self.radius = radius
        self.fading_rate = fading_rate
        self.delete_faded_threshold = delete_faded_threshold
        self.communication_step = communication_step
        self.alpha = alpha
        self.max_topic_count = max_topic_count
        self.outlier_threshold = outlier_threshold
        self.top_n = top_n
        self.clean_up_delta_time = clean_up_step
        self.data_agent = DataAgent(data_file_path=data_file_path, count=dp_count, is_twitter=is_twitter)
        self.generic_distance_function = generic_distance
        self.dp_id_to_agent_id = dict()
        self.global_idf_count = {}
        self.first_residual = None

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
        my_threads = []
        for agent_id in self.agents:
            t = Thread(target=self.agents[agent_id].get_outliers, args=[outliers_id])
            my_threads.append(t)
            t.daemon = True
            t.start()
        for t in my_threads:
            t.join()
        agents_to_remove = []
        for agent_id in self.agents:
            if len(self.agents[agent_id].dp_ids) < 1:
                agents_to_remove.append(agent_id)
        for aid in agents_to_remove:
            self.remove_agent(aid)

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
            for dp_id, distance, agent_id in outliers_to_join[self.top_n:]:
                del self.data_agent.data_points[dp_id]

        for dp_id, distance, agent_id in outliers_to_join:
            if distance > self.radius:
                new_agent_id = self.create_agent()
                self.agents[new_agent_id].add_data_point(self.data_agent.data_points[dp_id])
            else:
                self.agents[agent_id].add_data_point(self.data_agent.data_points[dp_id])

    def warm_up(self):
        for i in range(self.max_topic_count):
            self.create_agent()
        flag = True
        agents_dict = {id_: self.alpha for id_ in self.agents.keys()}
        for i in range(self.max_topic_count * self.alpha):
            random_agent_id = random.sample(list(agents_dict), k=1)[0]
            dp = self.data_agent.get_next_dp()
            if flag:
                self.first_residual = time.mktime(KingAgent.current_date.timetuple()) % get_seconds(
                    self.communication_step) - 1
                flag = False
            KingAgent.current_date = copy.deepcopy(dp.created_at)
            self.agents[random_agent_id].add_data_point(dp)
            agents_dict[random_agent_id] -= 1
            if agents_dict[random_agent_id] == 0:
                del agents_dict[random_agent_id]
        del agents_dict

    def stream(self, dp):
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

    def fade_agents_weight(self):
        for agent_id in list(self.agents.keys()):
            agent = self.agents[agent_id]
            agent.fade_agent_weight(self.fading_rate, self.delete_faded_threshold)

    def fade_agents_tfs(self):
        for agent_id in list(self.agents.keys()):
            agent = self.agents[agent_id]
            agent.fade_agent_tfs(self.fading_rate, self.delete_faded_threshold)
            # removing very small tf's happens in fade_agent_tfs too

    def handle_old_dps(self):
        for agent_id, agent in self.agents.items():
            agent.handle_old_dps()

    def train(self):
        self.warm_up()
        self.handle_outliers()
        KingAgent.dp_now = self.max_topic_count * self.alpha
        while self.data_agent.has_next_dp():
            dp = self.data_agent.get_next_dp()
            if (KingAgent.dp_now + 1) % 1000 == 0:
                print(f'data point count = {KingAgent.dp_now + 1} number of agents : {len(self.agents)}')
            KingAgent.dp_now += 1  # to count on what dp we are at now
            flag = True
            while flag:
                KingAgent.current_date += pd.Timedelta(seconds=1)
                if dp.created_at <= KingAgent.current_date:
                    self.stream(dp)
                    flag = False
                # communication
                residual = (time.mktime(KingAgent.current_date.timetuple()) - self.first_residual) % get_seconds(self.communication_step)
                if residual < KingAgent.prev_residual:
                    self.handle_old_dps()
                    self.handle_outliers()
                    self.fade_agents_weight()
                    # self.fade_agents_tfs()
                    print('cleaned up')
                KingAgent.prev_residual = residual
                print(residual)
                # save output every interval
                # save_output_residual = time.mktime(KingAgent.current_date.timetuple()) % get_seconds(self.save_output_interval)

                # if save_output_residual < KingAgent.save_output_prev_residual:
                #     self.handle_old_dps()
                #     self.handle_outliers()
                #     # self.fade_agents_weight()
                #     # self.fade_agents_tfs()
                #
                #     self.save_model(
                #         os.path.join(Path(os.getcwd()).parent, 'Data', 'outputs/multi_agent',
                #                      'X' + str(KingAgent.prev_full_date).replace(':', '_') + '--' + str(
                #                          KingAgent.dp_now),
                #                      'model'))
                #     self.write_output_to_files(
                #         os.path.join(Path(os.getcwd()).parent, 'Data', 'outputs/multi_agent',
                #                      'X' + str(KingAgent.prev_full_date).replace(':', '_') + '--' + str(
                #                          KingAgent.dp_now),
                #                      'clusters'))
                #     self.write_topics_to_files(
                #         os.path.join(Path(os.getcwd()).parent, 'Data', 'outputs/multi_agent',
                #                      'X' + str(KingAgent.prev_full_date).replace(':', '_') + '--' + str(
                #                          KingAgent.dp_now),
                #                      'topics'), 5)
                #     self.write_tweet_ids_to_files(
                #         os.path.join(Path(os.getcwd()).parent, 'Data', 'outputs/multi_agent',
                #                      'X' + str(KingAgent.prev_full_date).replace(':', '_') + '--' + str(
                #                          KingAgent.dp_now),
                #                      'clusters_tweet_ids'))
                #     print('saved')
                # KingAgent.prev_full_date = KingAgent.full_date
                # KingAgent.save_output_prev_residual = save_output_residual

        # self.fade_agents_weight()
        self.handle_old_dps()
        self.handle_outliers()
        # self.save_model(
        #     os.path.join(Path(os.getcwd()).parent, 'Data', 'outputs/multi_agent',
        #                  'X' + str(KingAgent.prev_full_date).replace(':', '_') + '--' + str(KingAgent.dp_now),
        #                  'model'))
        # self.write_output_to_files(
        #     os.path.join(Path(os.getcwd()).parent, 'Data', 'outputs/multi_agent',
        #                  'X' + str(KingAgent.prev_full_date).replace(':', '_') + '--' + str(KingAgent.dp_now),
        #                  'clusters'))
        # self.write_topics_to_files(
        #     os.path.join(Path(os.getcwd()).parent, 'Data', 'outputs/multi_agent',
        #                  'X' + str(KingAgent.prev_full_date).replace(':', '_') + '--' + str(KingAgent.dp_now),
        #                  'topics'), 5)
        # self.write_tweet_ids_to_files(
        #     os.path.join(Path(os.getcwd()).parent, 'Data', 'outputs/multi_agent',
        #                  'X' + str(KingAgent.prev_full_date).replace(':', '_') + '--' + str(KingAgent.dp_now),
        #                  'clusters_tweet_ids'))
        #
        # print('saved')

    def save_model(self, parent_dir):
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        with open(os.path.join(parent_dir, 'model.pkl'), 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_model(cls, file_dir):
        with open(file_dir, 'rb') as file:
            return pickle.load(file)

    def write_topics_to_files(self, parent_dir, max_topic_n=10):
        agent_topics = self.get_topics_of_agents(max_topic_n)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        for agent_id, topics in agent_topics.items():
            with open(os.path.join(parent_dir, f"{agent_id}.txt"), 'w', encoding='utf8') as file:
                for item in topics:
                    file.write(f'{self.data_agent.id_to_token[item[0]]} : {str(item[1])}\n')

    def write_output_to_files(self, parent_dir):
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        for agent_id, agent in self.agents.items():
            with open(os.path.join(parent_dir, f"{agent_id}.txt"), 'w', encoding='utf8') as file:
                for dp_id in agent.dp_ids:
                    dp_df = self.data_agent.raw_data.iloc[[self.data_agent.data_points[dp_id].index_in_df]]
                    if self.is_twitter:
                        file.write(str(dp_df['text'].values[0]) + '\n\n')
                    else:
                        file.write(str(dp_df['TEXT'].values[0]) + '\n\n')

    def write_tweet_ids_to_files(self, parent_dir):
        if not self.is_twitter:
            return
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        for agent_id, agent in self.agents.items():
            with open(os.path.join(parent_dir, f"{agent_id}.txt"), 'w', encoding='utf8') as file:
                for dp_id in agent.dp_ids:
                    tweet_id = self.data_agent.data_points[dp_id].status_id
                    if self.is_twitter:
                        file.write(str(tweet_id) + '\n')

    def get_topics_of_agents(self, max_topic_n=10):
        agent_topics = {}
        for agent_id, agent in self.agents.items():
            tf_idf = {}
            for term_id, f in agent.agent_f.items():
                dfi = 0
                for agent_id_2, agent_2 in self.agents.items():
                    if term_id in agent_2.agent_f:
                        dfi += 1

                tf_idf[term_id] = 1 + log((len(self.agents) + 1) / dfi) * (f / sum(agent.agent_f.values()))

            agent_topics[agent_id] = heapq.nlargest(max_topic_n, tf_idf.items(), key=lambda x: x[1])
        return agent_topics
