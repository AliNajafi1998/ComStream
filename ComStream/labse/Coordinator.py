import multiprocessing
import scipy
import pandas as pd
import random
import re
import time
import pickle
import os
from .DataManager import DataManager
from .Agent import Agent
from colorama import Fore
from .Utils import get_seconds
from sklearn.feature_extraction.text import TfidfVectorizer


class Coordinator:
    current_date = pd.to_datetime('2020-03-29T00:00:00Z')
    prev_date = pd.to_datetime('2020-03-28T00:00:00Z')

    dp_counter = 0

    def __init__(self,
                 save_output_interval: str,
                 init_no_agents: int,
                 communication_interval: str,
                 sliding_window_interval: str,
                 assign_radius: float,
                 init_dp_per_agent: int,
                 outlier_threshold: float,
                 dp_count: int,
                 no_topics: int,
                 no_keywords: int,
                 agent_fading_rate: float,
                 delete_agent_weight_threshold: float,
                 data_file_path: str,
                 embedding_file_path: str,
                 generic_distance,
                 data_start_date=pd.to_datetime('2020-03-29T00:00:00Z'),
                 is_parallel=True,
                 verbose=0
                 ):
        pattern = re.compile(r'^[0-9]+:[0-9]{2}:[0-9]{2}$')
        are_invalid_steps = len(pattern.findall(communication_interval)) != 1 or len(
            pattern.findall(sliding_window_interval)) != 1

        if are_invalid_steps:
            raise Exception(f'Invalid inputs fot steps')
        self.save_output_interval = save_output_interval
        self.agents = {}
        self.assign_radius = assign_radius
        self.agent_fading_rate = agent_fading_rate
        self.delete_agent_weight_threshold = delete_agent_weight_threshold
        self.communication_interval = communication_interval
        self.init_dp_per_agent = init_dp_per_agent
        self.init_no_agents = init_no_agents
        self.outlier_threshold = outlier_threshold
        self.no_keywords = no_keywords
        self.no_topics = no_topics
        self.sliding_window_interval = sliding_window_interval
        self.data_agent = DataManager(data_file_path=data_file_path, data_embedding_path=embedding_file_path,
                                      count=dp_count)
        self.generic_distance_function = generic_distance
        self.dp_id_to_agent_id = dict()
        self.first_communication_residual = None
        self.first_save_output_residual = None
        self.is_parallel = is_parallel
        self.verbose = verbose
        Coordinator.current_date = data_start_date
        Coordinator.prev_date = Coordinator.current_date + pd.Timedelta(days=-1)

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
        if self.is_parallel:
            cpu_count = multiprocessing.cpu_count() - 1
            agent_ids = list(self.agents.keys())
            start = 0
            while True:
                stop = start + cpu_count
                if stop < len(agent_ids):
                    self.parallel_outlier_getting(agent_ids, start, outliers_id, stop)
                else:
                    stop = len(agent_ids)
                    self.parallel_outlier_getting(agent_ids, start, outliers_id, stop)
                    break
                start = stop
        else:
            for agent_id in self.agents:
                self.agents[agent_id].get_outliers(outliers_id)

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
                distance = agent.get_distance(self, self.data_agent.data_points[outlier_id].embedding_vec)
                if distance <= min_distance:
                    min_distance = distance
                    similar_agent_id = agent_id
            if similar_agent_id != -1:
                outliers_to_join.append((outlier_id, min_distance, similar_agent_id))
            else:
                print('Sth went wrong!')

        for dp_id, distance, agent_id in outliers_to_join:
            # print(distance)
            if distance > self.assign_radius:
                new_agent_id = self.create_agent()
                self.agents[new_agent_id].add_data_point(self.data_agent.data_points[dp_id])
            else:
                self.agents[agent_id].add_data_point(self.data_agent.data_points[dp_id])

    def parallel_outlier_getting(self, agent_ids, index, outliers_id, stop):
        my_processes = []
        for agent_id in agent_ids[index:stop]:
            p = multiprocessing.Process(
                target=self.agents[agent_id].get_outliers, args=(outliers_id,))
            my_processes.append(p)
            p.daemon = True
            p.start()
        for p in my_processes:
            p.join()

    def init_agents(self):
        """
        filling the initial agents with initial dps at the start of the train
        :return: None
        """
        for i in range(self.init_no_agents):
            self.create_agent()
        flag = True
        agents_dict = {id_: self.init_dp_per_agent for id_ in self.agents.keys()}
        for i in range(self.init_no_agents * self.init_dp_per_agent):
            random_agent_id = random.sample(list(agents_dict), k=1)[0]
            dp = self.data_agent.get_next_dp()
            if flag:
                self.first_communication_residual = time.mktime(Coordinator.current_date.timetuple()) % get_seconds(
                    self.communication_interval) - 0
                self.first_save_output_residual = time.mktime(Coordinator.current_date.timetuple()) % get_seconds(
                    self.save_output_interval) - 0
                flag = False
            Coordinator.current_date = dp.created_at
            Coordinator.prev_date = dp.created_at
            self.agents[random_agent_id].add_data_point(dp)
            agents_dict[random_agent_id] -= 1
            if agents_dict[random_agent_id] == 0:
                del agents_dict[random_agent_id]
        del agents_dict
        if self.verbose == 1:
            print(f'Init_agents done : Number of agents : {len(self.agents)}')

    def stream(self, dp):
        """
        gets the dp, puts it in the closest agent if the agent is within the radius, else makes a new agent for it
        :param dp: the object of the dp
        :return: None
        """
        min_distance = float('infinity')
        similar_agent_id = -1
        for agent_id, agent in self.agents.items():
            distance = agent.get_distance(self, self.data_agent.data_points[dp.dp_id].embedding_vec)
            if distance <= min_distance:
                min_distance = distance
                similar_agent_id = agent_id
        if min_distance > self.assign_radius:
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
            agent.fade_agent_weight(self.agent_fading_rate, self.delete_agent_weight_threshold)

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
        self.init_agents()
        self.handle_outliers()
        Coordinator.dp_counter = self.init_no_agents * self.init_dp_per_agent
        while self.data_agent.has_next_dp():
            dp = self.data_agent.get_next_dp()
            if self.verbose != 0:
                if (Coordinator.dp_counter + 1) % 1000 == 0:
                    message = f'{Fore.CYAN}{Coordinator.current_date} : data point count = {Coordinator.dp_counter + 1}'
                    message += f' number of agents : {len(self.agents)}'
                    print(message)
            Coordinator.dp_counter += 1
            flag = True
            while flag:

                if dp.created_at != Coordinator.prev_date:
                    Coordinator.current_date += pd.Timedelta(seconds=1)

                    # communication every interval
                    self.communicate()

                    # save output every interval
                    self.save()

                if dp.created_at <= Coordinator.current_date:
                    self.stream(dp)
                    Coordinator.prev_date = dp.created_at
                    flag = False

        self.handle_old_dps()
        self.handle_outliers()
        self.save_model_and_files()

    def communicate(self):
        """
        checks if now is the right time to communicate, if so then handles outliers and old dps and fades agent weights
        :return: None
        """
        communication_residual = (time.mktime(
            Coordinator.current_date.timetuple()) - self.first_communication_residual) % get_seconds(
            self.communication_interval)
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
            Coordinator.current_date.timetuple()) - self.first_save_output_residual) % get_seconds(
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
        #     os.path.join(os.getcwd(), 'outputs/multi_agent',
        #                  'X' + str(Coordinator.current_date).replace(':', '_') + '--' + str(
        #                      Coordinator.dp_counter), 'model'))
        self.write_output_to_files(
            os.path.join(os.getcwd(), 'outputs/multi_agent',
                         'X' + str(Coordinator.current_date).replace(':', '_') + '--' + str(
                             Coordinator.dp_counter), 'clusters'))
        self.write_topics_to_files(
            os.path.join(os.getcwd(), 'outputs/multi_agent',
                         'X' + str(Coordinator.current_date).replace(':', '_') + '--' + str(
                             Coordinator.dp_counter), 'topics'))
        self.write_tweet_ids_to_files(
            os.path.join(os.getcwd(), 'outputs/multi_agent',
                         'X' + str(Coordinator.current_date).replace(':', '_') + '--' + str(
                             Coordinator.dp_counter), 'clusters_tweet_ids'))
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

    def write_topics_to_files(self, parent_dir):
        """
        write the max_topic_n topics to the file in parent_dir
        :param parent_dir: the directory of the parent where you want to write the topics at
        :return: None
        """
        topics_keywords = self.get_topics_of_agents()
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        topics_keywords_text = ''
        for keywords in topics_keywords:
            topics_keywords_text += ' '.join(keywords) + '\n'
        with open(os.path.join(parent_dir, f"top_topics_keywords.txt"), 'w', encoding='utf8') as file:
            file.write(topics_keywords_text.strip())

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
                    tweet = self.data_agent.data_points[dp_id].tweet
                    file.write(f'{tweet}\n\n')

    def write_tweet_ids_to_files(self, parent_dir):
        """
        for each agent a separate text file is generated which the id of dps of that agent are added to them
        :param parent_dir: the parent directory where you want the output to be at
        :return: None
        """
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        for agent_id, agent in self.agents.items():
            with open(os.path.join(parent_dir, f"{agent_id}.txt"), 'w', encoding='utf8') as file:
                for dp_id in agent.dp_ids:
                    tweet_id = self.data_agent.data_points[dp_id].status_id
                    file.write(str(tweet_id) + '\n')

    def get_topics_of_agents(self):
        topics_keywords = []
        full_corpus = []
        agent_ids_size = []  # (agent_id, size)
        for agent_id, agent_object in self.agents.items():
            agent_ids_size.append((agent_id, len(agent_object.dp_ids)))
            for dp_id in agent_object.dp_ids:
                dp_object = self.data_agent.data_points[dp_id]
                full_corpus.append(dp_object.tweet)
        vectorizer = TfidfVectorizer()
        # vectorized_corpus.shape (index of corpus, score of each word
        # (all the words evens words that don't exist in this sentence))
        vectorizer.fit(full_corpus)
        vocab = vectorizer.get_feature_names()  # ind -> word

        # get the biggest agents:
        agent_ids_size.sort(key=lambda x: x[1], reverse=True)
        candidate_agent_ids_size = agent_ids_size[:min(self.no_topics, len(agent_ids_size))]

        # get scores for each word in every agent
        for agent_id, agent_size in candidate_agent_ids_size:
            agent_corpus = []
            agent_object = self.agents[agent_id]
            for dp_id in agent_object.dp_ids:
                dp_object = self.data_agent.data_points[dp_id]
                agent_corpus.append(dp_object.tweet)
            vectorized_agent_corpus = vectorizer.transform(agent_corpus)

            cx = scipy.sparse.coo_matrix(vectorized_agent_corpus)
            agent_word2score = {}
            for i, j, v in zip(cx.row, cx.col, cx.data):
                agent_word2score[vocab[j]] = agent_word2score.get(vocab[j], 0.0) + v

            # sort and choose the top no_keywords
            sorted_keys = list(sorted(agent_word2score, key=agent_word2score.get, reverse=True))  # get the sorted keys
            sorted_keys = sorted_keys[:min(self.no_keywords, len(sorted_keys))]  # get best keywords
            topics_keywords.append(sorted_keys)

        del vocab
        del vectorizer
        return topics_keywords
