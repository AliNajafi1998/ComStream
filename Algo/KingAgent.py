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
from colorama import Fore


class KingAgent:
    current_date = pd.to_datetime('2020-03-29T00:00:00Z')
    prev_date = pd.to_datetime('2020-03-28T00:00:00Z')

    dp_counter = 0

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
                 max_keyword_per_agent: int,
                 fading_rate: float,
                 delete_faded_threshold: float,
                 data_file_path: str,
                 is_twitter=False,
                 generic_distance=get_distance_tf_idf_cosine,
                 verbose=0):
        """
        the class where every agent and dp is managed
        :param save_output_interval: the time interval in which we will save our agents, agents with tweet id's, models,
        topics
        :param max_topic_count: starting number of agents
        :param communication_step: the time interval in which the algorithm will communicate for deleting old dps,
        handle outliers, fading agents weight
        :param clean_up_step: the time interval in which the dp is considered an old dp
        :param radius: a dp is assigned to the closest agent if the distance is less than radius (1-similarity)
        :param alpha: number of initial dps per agent
        :param outlier_threshold: if in outlier detection a dps distance from agent is more than outlier_threshold,
        reassign the dp
        :param top_n: the maximum number of outliers we will re-assign in each clean up
        :param dp_count: the maximum number of dps to process in the algorithm
        :param fading_rate: the percentile of each agents weight that gets faded in each clean up
        :param delete_faded_threshold: in each clean up step, if any agents weight is less than this threshold,
        the agent gets deleted
        :param data_file_path: the path of the input data
        :param is_twitter: if the data is twitter True, else False
        :param generic_distance: the type of our distance metric
        :param verbose: 0 doesn't keep logs, 1 keeps logs
        :return: None
        """
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
        self.max_keyword_per_agent = max_keyword_per_agent
        self.clean_up_step = clean_up_step
        self.data_agent = DataAgent(data_file_path=data_file_path, count=dp_count, is_twitter=is_twitter)
        self.generic_distance_function = generic_distance
        self.dp_id_to_agent_id = dict()
        self.global_idf_count = {}
        self.first_communication_residual = None
        self.first_save_output_residual = None
        self.verbose = verbose

    def create_agent(self) -> int:
        """
        creates the agent and returns it's id
        :return: (int) returns the created agent's id
        """
        agent = Agent(self, generic_distance_function=self.generic_distance_function)
        self.agents[agent.agent_id] = agent
        return agent.agent_id

    def remove_agent(self, agent_id) -> None:
        """
        removes the agent with all it's dps
        :param agent_id: the id of the agent
        :return: None
        """
        for dp_id in self.agents[agent_id].dp_ids:
            self.agents[agent_id].remove_data_point(dp_id)
        del self.agents[agent_id]

    def handle_outliers(self) -> None:
        """
        looks at all the dps and if their distance from their assigned agent is more than outlier_threshold, then
        if there is another agent which is within the dps radius reassigns it, else creates a new agent for the dp
        :return: None
        """
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
        """
        filling the initial agents with initial dps at the start of the train
        :return: None
        """
        for i in range(self.max_topic_count):
            self.create_agent()
        flag = True
        agents_dict = {id_: self.alpha for id_ in self.agents.keys()}
        for i in range(self.max_topic_count * self.alpha):
            random_agent_id = random.sample(list(agents_dict), k=1)[0]
            dp = self.data_agent.get_next_dp()
            if flag:
                self.first_communication_residual = time.mktime(KingAgent.current_date.timetuple()) % get_seconds(
                    self.communication_step) - 0
                self.first_save_output_residual = time.mktime(KingAgent.current_date.timetuple()) % get_seconds(
                    self.save_output_interval) - 0
                flag = False
            KingAgent.current_date = dp.created_at
            KingAgent.prev_date = dp.created_at
            self.agents[random_agent_id].add_data_point(dp)
            agents_dict[random_agent_id] -= 1
            if agents_dict[random_agent_id] == 0:
                del agents_dict[random_agent_id]
        del agents_dict
        if self.verbose == 1:
            print(f'WarmUp done : Number of agents : {len(self.agents)}')

    def stream(self, dp):
        """
        gets the dp, puts it in the closest agent if the agent is within the radius, else makes a new agent for it
        :param dp: the object of the dp
        :return: None
        """
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
        """
        for each agent calls the function that will fade that agent's weight
        :return: None
        """
        for agent_id in list(self.agents.keys()):
            agent = self.agents[agent_id]
            agent.fade_agent_weight(self.fading_rate, self.delete_faded_threshold)

    def handle_old_dps(self):
        """
        for each agent, calls the function that handles the old dps
        :return: None
        """
        for agent_id, agent in self.agents.items():
            agent.handle_old_dps()

    def train(self):
        """
        the main training function that handles everything
        :return: None
        """
        self.warm_up()
        self.handle_outliers()
        KingAgent.dp_counter = self.max_topic_count * self.alpha
        while self.data_agent.has_next_dp():
            dp = self.data_agent.get_next_dp()
            if self.verbose != 0:
                if (KingAgent.dp_counter + 1) % 1000 == 0:
                    print(
                        f'{Fore.CYAN}{KingAgent.current_date} : data point count = {KingAgent.dp_counter + 1} number of agents : {len(self.agents)}')
            KingAgent.dp_counter += 1
            flag = True
            while flag:

                if dp.created_at != KingAgent.prev_date:
                    KingAgent.current_date += pd.Timedelta(seconds=1)

                    # communication every interval
                    self.communicate()

                    # save output every interval
                    self.save()

                if dp.created_at <= KingAgent.current_date:
                    self.stream(dp)
                    KingAgent.prev_date = dp.created_at
                    flag = False

        self.handle_old_dps()
        self.handle_outliers()
        self.save_model_and_files()

    def communicate(self):
        """
        checks if now is the right time to communicate, if so then handles outliers and old dps and fades agent weights
        :param agent_id: the id of the agent
        :return: None
        """
        communication_residual = (time.mktime(
            KingAgent.current_date.timetuple()) - self.first_communication_residual) % get_seconds(
            self.communication_step)
        if abs(communication_residual) <= 1e-7:
            self.handle_old_dps()
            self.handle_outliers()
            self.fade_agents_weight()
            if self.verbose != 0:
                print(f'{Fore.BLUE}{self.current_date} : Communicating -> Number of agents : {len(self.agents)}')

    def save(self):
        """
        check if now is the right time to save everything considering save_output_interval and call the function after
        :return: None
        """
        save_output_residual = (time.mktime(
            KingAgent.current_date.timetuple()) - self.first_save_output_residual) % get_seconds(
            self.save_output_interval)
        if abs(save_output_residual) <= 1e-7:
            self.handle_old_dps()
            self.handle_outliers()
            self.save_model_and_files()

    def save_model_and_files(self):
        """
        save the model, agent dps texts, agent dps ids, agent top topics
        :return: None
        """
        # self.save_model(
        #     os.path.join(Path(os.getcwd()).parent, 'Data', 'outputs/multi_agent',
        #                  'X' + str(KingAgent.current_date).replace(':', '_') + '--' + str(
        #                      KingAgent.dp_counter), 'model'))
        self.write_output_to_files(
            os.path.join(Path(os.getcwd()).parent, 'Data', 'outputs/multi_agent',
                         'X' + str(KingAgent.current_date).replace(':', '_') + '--' + str(
                             KingAgent.dp_counter), 'clusters'))
        self.write_topics_to_files(
            os.path.join(Path(os.getcwd()).parent, 'Data', 'outputs/multi_agent',
                         'X' + str(KingAgent.current_date).replace(':', '_') + '--' + str(
                             KingAgent.dp_counter), 'topics'), self.max_keyword_per_agent)
        self.write_tweet_ids_to_files(
            os.path.join(Path(os.getcwd()).parent, 'Data', 'outputs/multi_agent',
                         'X' + str(KingAgent.current_date).replace(':', '_') + '--' + str(
                             KingAgent.dp_counter), 'clusters_tweet_ids'))
        if self.verbose == 1:
            print(f'{Fore.YELLOW}{self.current_date} : Save Model and Outputs -> Number of agents : {len(self.agents)}')

    def save_model(self, parent_dir):
        """
        save the whole model at given directory
        :param parent_dir: the parent dir to store the model at
        :return: None
        """
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        with open(os.path.join(parent_dir, 'model.pkl'), 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_model(cls, file_dir):
        """
        load a saved model
        :param file_dir: the exact directory of the model you want to load
        :return: None
        """
        with open(file_dir, 'rb') as file:
            return pickle.load(file)

    def write_topics_to_files(self, parent_dir, max_topic_n=10):
        """
        write the max_topic_n topics to the file in parent_dir
        :param parent_dir: the directory of the parent where you want to write the topics at
        :param max_topic_n: how many of the top topics you want to write at the files
        :return: None
        """
        agent_topics = self.get_topics_of_agents(max_topic_n)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        for agent_id, topics in agent_topics.items():
            key_words_str = ''
            for item in topics:
                key_words_str += str(self.data_agent.id_to_token[item[0]]) + ' '
            with open(os.path.join(parent_dir, f"{agent_id}.txt"), 'w', encoding='utf8') as file:
                file.write(key_words_str)

    def write_output_to_files(self, parent_dir):
        """
        for each agent a separate text file is generated which the text of dps of that agent are added to them
        :param parent_dir: the parent directory where you want the output to be at
        :return: None
        """
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
        """
        for each agent a separate text file is generated which the id of dps of that agent are added to them
        :param parent_dir: the parent directory where you want the output to be at
        :return: None
        """
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
        """
        get max_topic_n topics (most frequent words of that agent) of each agent and return it
        :param max_topic_n: the maximum number of top topics you want to be returned
        :return: dict of list of tuple of top topics of each agent with their tf_idf value
        {agent_id: [(token_id1, tf_idf_value1), (token_id2, tf_idf_value2), ...]
        """
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
