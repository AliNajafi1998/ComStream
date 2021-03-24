import os


class Evaluate:
    def __init__(self, truth_dir, predicted_dir, top_n_topics, top_n_keywords, assigning_keyword_threshold):
        self.truth_dir = truth_dir
        self.predicted_dir = predicted_dir
        self.top_n_topics = top_n_topics
        self.top_n_keywords = top_n_keywords
        self.assigning_keyword_threshold = assigning_keyword_threshold

        self.empty_days = []
        self.predicted_topics = None  # list [day], list[topic], list[keyword_synonyms]
        self.truth_topics = None
        # for topic recall
        self.no_truth_topics = 0  # 'no_' means 'number of'
        self.no_found_truth_topics = 0
        # for keyword recall
        self.no_truth_keyword = 0
        self.no_found_truth_keywords = 0
        # for keyword precision
        self.no_prediction_keyword = 0
        self.no_found_prediction_keywords = 0

    def run(self):
        self.truth_topics = self.get_truth_topics()
        self.predicted_topics = self.get_predicted_topics()
        self.assign()
        self.topic_recall()
        self.keyword_precision()
        self.keyword_recall()
        self.f_score()

    def get_truth_topics(self):
        day_topics_keywords = []
        for day_file_name in os.listdir(self.truth_dir):
            topics_keywords = []
            day_topics_dir = os.path.join(self.truth_dir, day_file_name)
            # check if file is empty
            if os.stat(day_topics_dir).st_size == 0:
                self.empty_days.append(day_file_name)
                continue
            with open(day_topics_dir, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    keywords = []
                    is_in_brackets = False
                    synonyms = []
                    for word in line.split(','):
                        word = word.strip()
                        if word[0] == '[':
                            is_in_brackets = True
                        cleaned_word = self.remove_brackets_from_word(word)
                        if is_in_brackets:
                            synonyms.append(cleaned_word)
                        else:
                            keywords.append([cleaned_word])
                        if word[-1] == ']':
                            is_in_brackets = False
                            keywords.append(synonyms.copy())
                            synonyms.clear()
                    topics_keywords.append(keywords)
                day_topics_keywords.append(topics_keywords)
        return day_topics_keywords

    def get_predicted_topics(self):
        day_topics_keywords = []
        for day_file_name in os.listdir(self.predicted_dir):
            if day_file_name in self.empty_days:
                continue
            topics_keywords = []
            day_topics_dir = os.path.join(self.predicted_dir, day_file_name)
            with open(day_topics_dir, 'r') as f:
                lines = f.readlines()
                lines = lines[:self.top_n_topics]
                for line in lines:
                    keywords = [word for word in line.strip().split(' ')]
                    keywords = keywords[:self.top_n_keywords]
                    topics_keywords.append(keywords)
            day_topics_keywords.append(topics_keywords)
        return day_topics_keywords

    def assign(self):
        for ind_day in range(len(self.truth_topics)):
            for ind_true_topic in range(len(self.truth_topics[ind_day])):  # in true topic i

                self.no_truth_topics += 1
                ind_best_prediction = -1
                max_found_truth_keywords = 0
                max_correct_assigned_prediction_keywords = 0
                for ind_prediction_topic in range(len(self.predicted_topics[ind_day])):  # in pred topic j
                    found_truth_keywords = 0
                    correct_assigned_prediction_keywords = 0
                    found_truth_indexes = {}
                    for word_pred in self.predicted_topics[ind_day][ind_prediction_topic]:  # word in pred topic j
                        for ind_synonym_words_true, synonym_words_true in enumerate(
                                self.truth_topics[ind_day][ind_true_topic]):  # synonym words in true
                            if word_pred in synonym_words_true:
                                correct_assigned_prediction_keywords += 1
                                if ind_synonym_words_true not in found_truth_indexes:
                                    found_truth_keywords += 1
                                found_truth_indexes[ind_synonym_words_true] = 1
                    if (max_found_truth_keywords < found_truth_keywords) or (
                            (max_found_truth_keywords == found_truth_keywords) and (
                            max_correct_assigned_prediction_keywords < correct_assigned_prediction_keywords)):
                        ind_best_prediction = ind_prediction_topic
                        max_found_truth_keywords = found_truth_keywords
                        max_correct_assigned_prediction_keywords = correct_assigned_prediction_keywords
                if max_found_truth_keywords >= self.assigning_keyword_threshold:
                    self.no_found_truth_topics += 1

                    self.no_truth_keyword += len(self.truth_topics[ind_day][ind_true_topic])
                    self.no_found_truth_keywords += max_found_truth_keywords

                    self.no_prediction_keyword += len(self.predicted_topics[ind_day][ind_best_prediction]) - (
                            max_correct_assigned_prediction_keywords - max_found_truth_keywords)
                    self.no_found_prediction_keywords += max_found_truth_keywords
                else:
                    print(ind_day, self.truth_topics[ind_day])

        # print(self.no_found_truth_keywords, self.no_truth_keyword)
        # print(self.no_found_prediction_keywords, self.no_prediction_keyword)
        # print(self.no_found_truth_topics, self.no_truth_topics)

    def topic_recall(self):
        print('topic_recall  |  keyword_precision  |  keyword_recall  |  keyword_F_score')
        tr = self.no_found_truth_topics / self.no_truth_topics
        print("%.3f (%d/%d)" % (tr, self.no_found_truth_topics, self.no_truth_topics), end='  |  ')

    def keyword_precision(self):
        kp = self.no_found_prediction_keywords / self.no_prediction_keyword
        print("%.3f (%d/%d)" % (kp, self.no_found_prediction_keywords, self.no_prediction_keyword), end='  |  ')

    def keyword_recall(self):
        kr = self.no_found_truth_keywords / self.no_truth_keyword
        print("%.3f (%d/%d)" % (kr, self.no_found_truth_keywords, self.no_truth_keyword), end='  |  ')

    def f_score(self):
        kr = self.no_found_truth_keywords / self.no_truth_keyword
        kp = self.no_found_prediction_keywords / self.no_prediction_keyword

        print("%.3f " % ((2 * kr * kp) / (kp + kr)))

    @staticmethod
    def remove_brackets_from_word(word):
        if word[0] == '[':
            word = word[1:]
        if word[-1] == ']':
            word = word[:-1]
        return word


if __name__ == '__main__':
    # 'C:/Users/shila/Desktop/covid-stream/Data/outputs/other_topics/LDA'
    # 'C:/Users/shila/Desktop/covid-stream/Data/outputs/multi_agent_topics'
    # for x in range(20):
    #     print(x+1,end=':\n')
    x = 19
    EV = Evaluate(truth_dir='C:/Users/shila/Desktop/covid-stream/Data/corona_truth',
                  predicted_dir='C:/Users/shila/Desktop/covid-stream/Data/outputs/multi_agent_topics',
                  top_n_topics=x + 1,
                  top_n_keywords=5,
                  assigning_keyword_threshold=2
                  # the pred_topic needs at least this many matched keywords for this topic to be considered found
                  )
    EV.run()
