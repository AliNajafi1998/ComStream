from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import glob

os.chdir('..')
path = os.getcwd()
df = pd.read_pickle(os.path.join(path, 'Data/reuters_cleaned.pkl')).head(10000)
os.chdir('./DBScan')

docs = df['TEXT'].tolist()
vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(docs)
db = DBSCAN(eps=0.70, metric='cosine', min_samples=6).fit(X)  # 70, 6
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print(n_clusters_)

files = glob.glob(str(os.getcwd()) + '/output/*')
for f in files:
    os.remove(f)

for index, label in enumerate(labels):
    with open(os.path.join(os.getcwd(), f'output/{label}.txt'), 'a') as f:
        f.write(df.iloc[[index]]['TEXT'].values[0])
        f.write('\n')
