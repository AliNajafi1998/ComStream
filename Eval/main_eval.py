import os
import pandas as pd
from pathlib import Path
from ACC import acc, nmi, ari, purity_score
import numpy as np
from collections import Counter

path = Path(os.getcwd())

# assign the data dirs
true_dir = str(path.parent) + '/Data/reuters_cleaned.pkl'
pred_dir = str(path.parent) + '/Algo/last_output/output'

# read the true and pred data
df_true_data = pd.read_pickle(true_dir)
pred_text2label = {}  # dp_text -> label

no_clusters = 0
no_clusters_without_outliers = 0
number_of_dps_with_outliers = 0
number_of_dps_without_outliers = 0
for label_dir in os.listdir(pred_dir):
    f = open(pred_dir + '/' + label_dir, "r")  # , encoding='utf8')
    lines = f.readlines()
    clean_lines = []
    all_text = ''
    for line in lines:
        all_text += line
    lines = [line.strip() for line in all_text.split('\n\n\n')]
    for line in lines:
        if line != '':
            clean_lines.append(line)

    number_of_dps_with_outliers += len(lines)
    no_clusters += 1
    # if len(lines) < 100:
    #     continue
    no_clusters_without_outliers += 1
    number_of_dps_without_outliers += len(lines)
    for line in lines:
        pred_text2label[line.strip()] = int(label_dir[:-4])
    f.close()
# assign each topic an index, dict: topic -> label
true_topic2ind = {}
count = 0
for index, row in df_true_data.iterrows():
    # if row['TOPICS'] == '' or ',' in row['TOPICS']:
    #     continue
    if row['TOPICS'] not in true_topic2ind:
        true_topic2ind[row['TOPICS']] = count
        count += 1
# iterate through true
y_pred, y_true = [], []
for index, row in df_true_data.iterrows():
    tweet_text = row['TEXT'].strip()
    # if row['TOPICS'] == '' or ',' in row['TOPICS'] or tweet_text not in pred_text2label:
    #     continue
    if tweet_text not in pred_text2label:
        continue
    y_true.append(true_topic2ind[row['TOPICS']])
    y_pred.append(pred_text2label[tweet_text])

print(f'number of all clusters:{no_clusters}')
print(f'number of all without the outliers: {no_clusters_without_outliers}')
print(f'number of all dps: {number_of_dps_with_outliers}')
print(f'number of all dps without the outliers: {number_of_dps_without_outliers}')
print(f'size of y_pred after getting rid of outliers: {len(y_pred)}')
print(f'size of y_true after getting rid of outliers: {len(y_true)}')

print(len(Counter(y_pred).keys()))
print(len(Counter(y_true).keys()))

print(f'Purity: {purity_score(np.array(y_true), np.array(y_pred))}')
print(f'ACC: {acc(np.array(y_true), np.array(y_pred))}')
print(f'NMI: {nmi(np.array(y_true), np.array(y_pred))}')
print(f'ARI: {ari(np.array(y_true), np.array(y_pred))}')
