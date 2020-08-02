import os
import pandas as pd
from pathlib import Path
from ACC import acc, nmi, ari
import numpy as np
from collections import Counter

path = Path(os.getcwd())

# assign the data dirs
true_dir = str(path.parent) + '/Data/reuters_cleaned.pkl'
pred_dir = str(path.parent) + '/Data/output/'

# read the true and pred data
df_true_data = pd.read_pickle(true_dir)
pred_text2label = {} # dp_text -> label
for label_dir in os.listdir(pred_dir):
    f = open(pred_dir+'/'+label_dir, "r")
    for line in f.readlines():
        pred_text2label[line.strip()] = int(label_dir[:-4])
    f.close()
# assign each topic an index, dict: topic -> label
true_topic2ind = {}
count = 0
for index, row in df_true_data.iterrows():
    if row['TOPICS'] == '' or ',' in row['TOPICS']:
        continue
    if row['TOPICS'] not in true_topic2ind:
        true_topic2ind[row['TOPICS']] = count
        count += 1

# iterate through true
y_pred, y_true = [], []
for index, row in df_true_data.iterrows():
    tweet_text = row['TEXT'].strip()
    if row['TOPICS'] == '' or ',' in row['TOPICS'] or tweet_text not in pred_text2label:
        continue
    y_true.append(true_topic2ind[row['TOPICS']])
    y_pred.append(pred_text2label[tweet_text])


print(len(y_pred))
print(len(y_true))
print(len(Counter(y_pred).keys()))
print(len(Counter(y_true).keys()))
# print(acc(np.array(y_true), np.array(y_pred)))
print(acc(np.array(y_true), np.array(y_pred)))
print(nmi(np.array(y_true), np.array(y_pred)))
print(ari(np.array(y_true), np.array(y_pred)))

