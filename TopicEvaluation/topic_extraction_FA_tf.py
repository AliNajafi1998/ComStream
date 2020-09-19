import os
from pathlib import Path
import spacy
import pytextrank
import pandas as pd
import gensim
from datetime import datetime
from gensim import corpora, models


# pip install pytextrank
# python -m spacy download en_core_web_sm
# pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz

def get_pred_data(out_dirs, top_n):
    days_list = []
    days_agent_texts = []  # list (day) of, list (size_agent, text_agent)
    for out_dir in os.listdir(out_dirs):
        if out_dir in ['X2012-05-05 16_27_00--22154', 'X2012-05-05 16_42_00--32634']:
            continue
        days_list.append(out_dir)
        one_day_dir = os.path.join(out_dirs, out_dir, 'clusters')
        texts = []
        for label_name in os.listdir(one_day_dir):
            f = open(os.path.join(one_day_dir, label_name), "r")  # , encoding='utf8')
            lines = f.readlines()
            text = ''
            count = 0
            for line in lines:
                if line.strip() == '':
                    continue
                text += line.strip() + '. '
                # text += line.strip() + ' '
                count += 1
            texts.append((count, text.strip(), label_name))
        texts.sort(reverse=True)
        texts = texts[:min(top_n, len(texts))]
        days_agent_texts.append(texts)

    # days_list, days_agent_texts = zip(*sorted(zip(days_list, days_agent_texts)))
    # print(days_list)
    # print(days_agent_texts)
    return days_agent_texts, days_list


def get_topics_top_tf(text2, top_n):
    text2 = text2.replace(".", "")
    word2count = {}
    for word in text2.split():
        word2count[word] = word2count.get(word, 0) + 1

    keywords2 = set({})
    for word, count in sorted(word2count.items(), key=lambda item: item[1], reverse=True):
        keywords2.add(word)
        if len(keywords2) == top_n:
            break
    return keywords2
    # return ranks

def get_topics_top_tf_idf(text2, top_n):
    text2 = text2.replace(".", "")
    word2count = {}
    for word in text2.split():
        word2count[word] = word2count.get(word, 0) + 1

    keywords2 = set({})
    for word, count in sorted(word2count.items(), key=lambda item: item[1], reverse=True):
        keywords2.add(word)
        if len(keywords2) == top_n:
            break
    return keywords2


if __name__ == '__main__':
    time_range = 5
    no_keywords = 5
    no_topics = 40
    multiagent_days_topics_keywords = []
    out_dirs = os.path.join(Path(os.getcwd()).parent, 'Data/outputs/multi_agent')
    # list (day) of, list (size_agent, text_agent, label_file_name)
    data, days_list = get_pred_data(out_dirs=out_dirs, top_n=no_topics)
    print('got the data')
    for day_ind, texts in enumerate(data):
        topics_keywords = []

        ########################## do the time stuff
        time = days_list[day_ind]
        dt = str(time)[12:20]
        date_time_obj = datetime.strptime(dt, '%H_%M_%S')
        date_time_obj -= pd.Timedelta(seconds=time_range * 60)
        topic_name = str(date_time_obj)[11:13] + '_' + str(date_time_obj)[14:16]
        if topic_name[-2] == '0':
            topic_name = topic_name[:len(topic_name) - 2] + topic_name[-1]

        # print(str(date_time_obj))
        list_dates_to_save = ['16_16', '16_26', '16_41', '16_53', '17_1', '17_3', '17_18', '17_24', '17_25', '17_36',
                              '17_46', '17_56', '18_9']
        if topic_name not in list_dates_to_save:
            continue
        print(topic_name)

        ############################### get labels
        for count, text, label_name in texts:
            # key_words = get_topics_textrank(text, top_n=no_keywords)
            key_words = get_topics_top_tf(text, top_n=no_keywords)
            keywords = list(key_words)
            topics_keywords.append(keywords)
        multiagent_days_topics_keywords.append(topics_keywords)
        topics_keywords_str = ''
        for keywords in topics_keywords:
            for keyword in keywords:
                topics_keywords_str += keyword + ' '
            topics_keywords_str = topics_keywords_str.strip()
            topics_keywords_str += '\n'

        with open(f'C:/Users/shila/Desktop/topics/pred/5_5_2012_{topic_name}.txt',
                  'w') as f:
            f.write(topics_keywords_str)

    # # returns tuple (str day, list of list [[topic1_key1,topic1_key2], [topic2_key1,topic2_key2]])
    # lda_days_topics_keywords = LDA_IDF(num_topics=20, max_keywords=5, time_range=time_range)
    # save_lda_results(lda_days_topics_keywords)
