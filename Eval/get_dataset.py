import random
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def make_reuters_data(data_dir):
    np.random.seed(1234)
    from sklearn.feature_extraction.text import CountVectorizer
    from os.path import join
    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            if cat in cat_list:
                did_to_cat[did] = did_to_cat.get(did, []) + [cat]
        # did_to_cat = {k: did_to_cat[k] for k in list(did_to_cat.keys()) if len(did_to_cat[k]) > 1}
        for did in list(did_to_cat.keys()):
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]

    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
        with open(join(data_dir, dat)) as fin:
            for line in fin.readlines():
                if line.startswith('.I'):
                    if 'did' in locals():
                        assert doc != ''
                        if did in did_to_cat:
                            data.append(doc)
                            target.append(cat_to_cid[did_to_cat[did][0]])
                    did = int(line.strip().split(' ')[1])
                    doc = ''
                elif line.startswith('.W'):
                    assert doc == ''
                else:
                    doc += line

    print((len(data), 'and', len(did_to_cat)))
    print(data[0])
    assert len(data) == len(did_to_cat)
    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
    y = np.asarray(target)
    from sklearn.feature_extraction.text import TfidfTransformer
    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
    x = x[:10000].astype(np.float32)
    print(x.dtype, x.size)
    y = y[:10000]
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
    print('todense succeed')

    p = np.random.permutation(x.shape[0])
    x = x[p]
    y = y[p]
    print('permutation finished')

    assert x.shape[0] == y.shape[0]
    x = x.reshape((x.shape[0], -1))
    np.save(join(data_dir, 'reutersidf10k.npy'), {'data': x, 'label': y})


date_time = '2000-01-01T00:00:01Z'


def update_time():
    global date_time
    d, t = date_time[0:-1].split('T')
    h, m, s = t.split(':')
    y, M, D = d.split('-')

    y = int(y)
    M = int(M)
    D = int(D)
    h = int(h)
    m = int(m)
    s = int(s)

    s += 1
    if s == 60:
        s = 0
        m += 1
        if m == 60:
            m = 0
            h += 1
            if h == 24:
                h = 0
                D += 1
                if D == 31:
                    D = 1
                    M += 1
                    if M == 13:
                        M = 1
                        y += 1
    y = str(y)
    M = str(M)
    D = str(D)
    h = str(h)
    m = str(m)
    s = str(s)
    if len(s) == 1:
        s = '0' + s
    if len(m) == 1:
        m = '0' + m
    if len(h) == 1:
        h = '0' + h
    if len(D) == 1:
        D = '0' + D
    if len(M) == 1:
        M = '0' + M

    date_time = f'{y}-{M}-{D}T{h}:{m}:{s}Z'


def make_reuters_data_with_time(data_dir, n_dp):
    np.random.seed(1234)
    from sklearn.feature_extraction.text import CountVectorizer
    from os.path import join
    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            if cat in cat_list:
                did_to_cat[did] = did_to_cat.get(did, []) + [cat]
        # did_to_cat = {k: did_to_cat[k] for k in list(did_to_cat.keys()) if len(did_to_cat[k]) > 1}
        for did in list(did_to_cat.keys()):
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]

    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
        with open(join(data_dir, dat)) as fin:
            for line in fin.readlines():
                if line.startswith('.I'):
                    if 'did' in locals():
                        assert doc != ''
                        if did in did_to_cat:
                            data.append(doc)
                            target.append(cat_to_cid[did_to_cat[did][0]])
                    did = int(line.strip().split(' ')[1])
                    doc = ''
                elif line.startswith('.W'):
                    assert doc == ''
                else:
                    doc += line

    print((len(data), 'and', len(did_to_cat)))
    assert len(data) == len(did_to_cat)

    # data: a list of text

    main_df = pd.DataFrame()
    main_df['TEXT'] = data  # x
    main_df['TOPICS'] = np.asarray(target)  # y
    # shuffle all the data
    p = np.random.permutation(len(main_df))
    main_df = main_df.iloc[p]
    print('permutation finished')
    main_df = main_df.iloc[:n_dp]  # choose only 10000 dp
    main_df.reset_index()

    counter = 0
    # while counter < len(main_df):
    #     if counter%1000==0:
    #         print(counter)
    #     random_number = random.randint(1, 10)
    #     if counter + random_number > len(main_df):
    #         random_number = len(main_df) - counter
    #     for j in range(counter, counter + random_number):
    #         main_df.loc[j, 'CREATED_AT'] = date_time
    #         counter += 1
    #     update_time()

    times = []
    for counter in range(len(main_df)):
        times.append(date_time)
        update_time()
    main_df['CREATED_AT'] = times
    main_df.reset_index(inplace=True)
    main_df.drop(columns=['index'])
    main_df.to_pickle("../Data/reuters_cleaned.pkl")


def load_reuters(n_dp, data_path='./reuters_raw_data'):
    import os
    if not os.path.exists(os.path.join(data_path, 'reutersidf10k.npy')):
        make_reuters_data_with_time(data_path, n_dp)
        print(('reutersidf saved to ' + data_path))
    data = np.load(os.path.join(data_path, 'reutersidf10k.npy'), allow_pickle=True)
    data_dict = data.item()
    # data = data.items()
    # has been shuffled
    x = data_dict['data']
    y = data_dict['label']
    x = x.reshape((x.shape[0], -1)).astype('float64')
    y = y.reshape((y.size,))
    print(f'REUTERSIDF10K samples')
    print(f'x_shape: {x.shape}, y_shape: {y.shape}')
    return x, y


if __name__ == '__main__':
    make_reuters_data_with_time(data_dir='../Data/reuters_raw_data', n_dp=10000)
    # x, y = load_reuters(n_dp=10000)
