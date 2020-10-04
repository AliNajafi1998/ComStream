import os
from pathlib import Path
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime


class TopicExtractor:
    def __init__(self, data_dir, save_dir, top_n_topics=10, top_n_keywords=10):
        self.save_dir = save_dir
        self.top_n_topics = top_n_topics
        self.top_n_keywords = top_n_keywords
        self.out_days_dir = data_dir
        self.days = []  # days names
        self.vectorizer = None  # we use sklearn for tfidf
        self.ind2word = None

    def extract_and_save_in_files(self):
        days_clusters = self.get_multi_agent_output()
        self.get_tf_idf(days_clusters)
        days_topics_keywords = self.get_topics_keywrods(days_clusters, self.top_n_topics, self.top_n_keywords)
        self.save_topics_in_files(days_topics_keywords)

    def get_multi_agent_output(self):
        days_clusters = []  # list (day) of list( tuple(cluster_len, texts))
        for day_file_name in os.listdir(self.out_days_dir):
            print(self.get_previous_day_date(day_file_name[1:11]))
            self.days.append(day_file_name[1:11])
            clusters = []  # (list of texts(dps))
            out_day_dir = os.path.join(self.out_days_dir, day_file_name, 'clusters')
            for cluster_file_name in os.listdir(out_day_dir):
                cluster_dir = os.path.join(out_day_dir, cluster_file_name)
                with open(cluster_dir, "r") as f:  # , encoding='utf8')
                    lines = f.readlines()
                    clean_lines = []
                    for line in lines:
                        if line.strip() != '':
                            clean_lines.append(line.strip())
                    clusters.append((len(clean_lines), clean_lines))
            clusters.sort(reverse=True)
            days_clusters.append(clusters)
        return days_clusters

    def get_tf_idf(self, days_clusters):
        corpus = []
        for clusters in days_clusters:
            for (cluster_size, cluster_lines) in clusters:
                cluster_concatenated_texts = ''
                for line in cluster_lines:
                    cluster_concatenated_texts += line + ' '
                corpus.append(cluster_concatenated_texts.strip())
        self.vectorizer = TfidfVectorizer(analyzer='word', tokenizer=lambda x: x.split())
        self.vectorizer.fit(corpus)
        self.ind2word = self.vectorizer.get_feature_names()

    def get_topics_keywrods(self, days_clusters, top_n_topics=10, top_n_keywords=10):
        days_topics_keywords = []
        for clusters in days_clusters:
            topics_keywords = []
            top_clusters = clusters[:top_n_topics]  # had sorted it before
            for (cluster_size, cluster_lines) in top_clusters:
                cluster_concatenated_texts = ''
                for line in cluster_lines:
                    cluster_concatenated_texts += line + ' '
                one_cluster_bow = self.vectorizer.transform([cluster_concatenated_texts.strip()])
                one_cluster_bow = one_cluster_bow.toarray()[0]  # just a bit of reshaping
                list_tf = []
                for i in range(one_cluster_bow.shape[0]):
                    if one_cluster_bow[i] != 0:
                        list_tf.append((one_cluster_bow[i], self.ind2word[i]))  # (freq, term)
                list_tf.sort(reverse=True)
                list_tf = list_tf[:top_n_keywords]
                topic_keywords = [keyword for (freq, keyword) in list_tf]
                topics_keywords.append(topic_keywords)
            days_topics_keywords.append(topics_keywords)
        return days_topics_keywords

    def save_topics_in_files(self, days_topics_keywords):
        for day_ind, topics_keywords in enumerate(days_topics_keywords):
            topics_text = ''
            for keywords in topics_keywords:
                keywords_text = ''
                for keyword in keywords:
                    keywords_text += keyword + ' '
                keywords_text = keywords_text.strip(' ')
                topics_text += keywords_text + '\n'
            file_name = self.get_previous_day_date(self.days[day_ind]) + '.txt'
            self.save_in_file(parent_dir=self.save_dir, file_name=file_name, text=topics_text)

    def save_in_file(self, parent_dir, file_name, text):
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        with open(os.path.join(parent_dir, file_name), 'w', encoding='utf8') as file:
            file.write(text)

    def get_previous_day_date(self, cur_date):
        my_date = datetime.strptime(cur_date, "%Y-%m-%d")
        delta_time = timedelta(days=1)
        my_date -= delta_time
        my_date = str(my_date)[:10]
        return my_date


if __name__ == '__main__':
    in_days_dir = os.path.join(Path(os.getcwd()).parent, 'Data/outputs/multi_agent')
    out_save_dir = os.path.join(Path(os.getcwd()).parent, 'Data/outputs/multi_agent_topics')
    TE = TopicExtractor(data_dir=in_days_dir, save_dir=out_save_dir, top_n_topics=20, top_n_keywords=20)
    TE.extract_and_save_in_files()
