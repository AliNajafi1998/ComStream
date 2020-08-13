from sklearn.cluster import KMeans
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

os.chdir('..')
path = os.getcwd()
df = pd.read_pickle(os.path.join(path, 'Data/data_cleaned.pkl')).head(10000)
os.chdir('./KMeans')

docs = []
for i in range(len(df)):
    docs.append(df.iloc[[i]]['text'].values[0])

vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(docs)

kmeans = KMeans(10).fit(X)

for i in range(10000):
    file_name = kmeans.predict(X[i])[0]
    with open(os.path.join(os.getcwd(), f'output/{file_name}.txt'), 'a') as f:
        f.write(docs[i])
        f.write('\n')
