from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

os.chdir('..')
path = os.getcwd()
df = pd.read_pickle(os.path.join(path, 'Data/data_cleaned.pkl')).head(10000)
os.chdir('./DBScan')

docs = []
for i in range(len(df)):
    docs.append(df.iloc[[i]]['text'].values[0])

vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(docs)
db = DBSCAN(eps=0.65, metric='cosine', min_samples=10).fit(X)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print(n_clusters_)

for index, label in enumerate(labels):
    with open(os.path.join(os.getcwd(), f'output/{label}.txt'), 'a') as f:
        f.write(df.iloc[[index]]['text'].values[0])
        f.write('\n')
