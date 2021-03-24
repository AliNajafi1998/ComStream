import pandas as pd
from preprocessingfacup import preprocess

df = pd.read_pickle('C:/Users/shila/Desktop/GitProjects/a_covid2/ComStream/Data/FA_Preprocessed.pkl').reset_index()

print(df.columns)
cnt = 0

for index, row in df.iterrows():
    txt = row['text']
    txt = preprocess(txt).strip()
    words = txt.split()
    for word in words:
        if word.strip() == 'lampard':
            cnt += 1
    if index % 1000 == 0:
        print(row['created_at'], cnt)
