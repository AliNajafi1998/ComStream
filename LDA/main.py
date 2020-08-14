import gensim
import numpy as np
import pandas as pd
import os
from gensim import corpora, models
import re

np.random.seed(2019)


def X():
    os.chdir('..')
    path = os.getcwd()
    df = pd.read_pickle(os.path.join(path, 'Data/reuters_cleaned.pkl'))
    os.chdir('./LDA')

    processed_docs = pd.Series(dtype=object)
    for i in range(len(df)):
        tokens = df.iloc[i]['TEXT'].split()
        tokens = list(filter(lambda x: x != "", tokens))
        processed_docs.at[i] = tokens

    dictionary = gensim.corpora.Dictionary(processed_docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=100, id2word=dictionary, passes=2, workers=4)

    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))

    count = 0
    for dp in processed_docs:
        unseen_document = dp
        bow_vector = dictionary.doc2bow(unseen_document)
        index = sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1 * tup[1])[0][0]
        # for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1 * tup[1]):
        pattern = r"\"[a-z0-9]+\""
        topic_name = re.findall(pattern, lda_model_tfidf.print_topic(index, 1))[0][1:-1]
        with open(os.path.join(os.getcwd(), f'output/{topic_name}.txt'), 'a') as f:
            count += 1
            f.write(' '.join(dp))
            f.write('\n\n')
    print(count)


if __name__ == '__main__':
    X()
