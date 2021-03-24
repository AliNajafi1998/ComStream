import os
from pathlib import Path
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import pandas as pd

data = pd.read_pickle("C:/Users/shila/Desktop/covid-stream/Data/FA/FACUP.pkl")
data['created_at'] = data['created_at'].apply(lambda x: str(x))

data['created_at'] = data['created_at'].apply(lambda x: str(x)[11:16])
print(data['created_at'].unique())

data = data.loc[data['created_at'] == '16:26']
print(data['created_at'].unique())
n_print = 20
texts = {}
for index, row in data.iterrows():
    txt = row['text']
    if txt in texts:
        continue
    texts[txt] = 1
    if 'ramires' in txt.split(' ') and '1-0' in txt.split(' ') :#and 'congrats' in txt.split(' '):
        print(row['status_id'], txt)
        n_print -= 1
        if n_print == 0:
            break
