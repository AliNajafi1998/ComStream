import os
from pathlib import Path
import spacy
import pytextrank
import pandas as pd
import numpy as np
import gensim
from gensim import corpora, models


# pip install pytextrank
# python -m spacy download en_core_web_sm
# pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz

def get_pred_data(out_dirs, top_n):
    days_list = []
    days_agent_texts = []  # list (day) of, list (size_agent, text_agent)
    for out_dir in os.listdir(out_dirs):
        days_list.append(out_dir[1:11])
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
                count += 1
            texts.append((count, text.strip(), label_name))
        texts.sort(reverse=True)
        texts = texts[:min(top_n, len(texts))]
        days_agent_texts.append(texts)

    # days_list, days_agent_texts = zip(*sorted(zip(days_list, days_agent_texts)))
    # print(days_list)
    # print(days_agent_texts)
    return days_agent_texts, days_list


def get_topic(text, top_n):
    # load a spaCy model, depending on language, scale, etc.
    nlp = spacy.load("en_core_web_sm")

    # add PyTextRank to the spaCy pipeline
    tr = pytextrank.TextRank()
    nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

    doc = nlp(text)

    # examine the top-ranked phrases in the document
    ranks = []
    keywords = set({})
    # for sent in doc._.textrank.summary(limit_phrases=15, limit_sentences=5):
    #     print(sent, end='***\n\n')
    for p in doc._.phrases:
        # print("{:.4f} {:5d}  {}".format(p.rank, p.count, p.text))
        ranks.append((p.rank, p.count, p.text))
        for word in p.text.split():
            keywords.add(word)
            if len(keywords) == top_n:
                break
        if len(keywords) == top_n:
            break
        # if len(ranks) == top_n:
        #     break
    return keywords
    # return ranks


def LDA_IDF(num_topics):
    os.chdir('..')
    path = os.getcwd()
    df = pd.read_pickle(os.path.join(path, 'Data/data_cleaned_1000k.pkl'))

    date2inds = {}  # date to list of tweet inds in the df
    for i in range(len(df)):
        date = df.iloc[i]['created_at']
        date = date[:10]
        if date not in date2inds:
            date2inds[date] = []
        date2inds[date].append(i)
    days_topics_keywords = []
    for date, inds in sorted(date2inds.items(), key=lambda x: x[1]):
        df2 = df.iloc[inds]
        processed_docs = pd.Series(dtype=object)
        for i in range(len(df2)):
            tokens = df2.iloc[i]['text'].split()
            tokens = list(filter(lambda x: x != "", tokens))
            processed_docs.at[i] = tokens

        dictionary = gensim.corpora.Dictionary(processed_docs)
        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]
        lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=2,
                                                     workers=4)
        topics_keywords = []
        for idx, topic in lda_model_tfidf.print_topics(-1):
            # print('Topic: {} Word: {}'.format(idx, topic))
            keywords = []
            for ind2, keyword in enumerate(topic.split('"')):
                if ind2 % 2 == 1:
                    keywords.append(keyword)
            topics_keywords.append(keywords)
        days_topics_keywords.append(topics_keywords)
        return topics_keywords
    return days_topics_keywords
    #
    # count = 0
    # for dp in processed_docs:
    #     unseen_document = dp
    #     bow_vector = dictionary.doc2bow(unseen_document)
    #     index = sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1 * tup[1])[0][0]
    #     # for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1 * tup[1]):
    #     pattern = r"\"[a-z0-9]+\""
    #     topic_name = re.findall(pattern, lda_model_tfidf.print_topic(index, 1))[0][1:-1]
    #     with open(os.path.join(os.getcwd(), f'output/{topic_name}.txt'), 'a') as f:
    #         count += 1
    #         f.write(' '.join(dp))
    #         f.write('\n\n')
    # print(count)


if __name__ == '__main__':
    multiagent_days_topics_keywords = []
    out_dirs = os.path.join(Path(os.getcwd()).parent, 'Data/outputs/multi_agent')
    data, days_list = get_pred_data(out_dirs=out_dirs, top_n=10)  # list (day) of, list (size_agent, text_agent)
    for day_ind, texts in enumerate(data):
        topics_keywords = []
        for count, text, label_name in texts:
            # ranks = get_topics(text, top_n=5)
            # print(days_list[day_ind], label_name)
            # for p in ranks:
            #     print("{:.4f} {:5d}  {}".format(p[0], p[1], p[2]))
            key_words = get_topic(text, top_n=5)
            keywords = list(key_words)
            multiagent_days_topics_keywords.append(keywords)
            # break
        break

    lda_days_topics_keywords = LDA_IDF(num_topics=5)

    truth = ['state', 'numbers', 'united', 'states', 'leader', 'infected', 'die', 'selfish', 'idiots']

    for topic in multiagent_days_topics_keywords:
        for keyword in topic:
            if keyword in truth:
                print(keyword)

    print('nextttttttt')
    for topic in lda_days_topics_keywords:
        for keyword in topic:
            if keyword in truth:
                print(keyword)