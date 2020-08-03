from sklearn.manifold import TSNE
import os
import pandas as pd
from pathlib import Path
from ACC import acc, nmi, ari
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

path = Path(os.getcwd())

# assign the data dirs
true_dir = str(path.parent) + '/Data/reuters_cleaned.pkl'
pred_dir = str(path.parent) + '/Data/output/'

# read the true and pred data
word2ind = {}
ind_n = 0  # keep track of the number of new words in word2ind
dp_n = 0  # keep track of overall number of dps we have in the file
# fill the dict word2ind
for label_dir in os.listdir(pred_dir):
    f = open(pred_dir + '/' + label_dir, "r")
    for line in f.readlines():
        dp_n += 1
        for word in line.split():
            if word not in word2ind:
                word2ind[word] = ind_n
                ind_n += 1
    f.close()

# getting the real data to bag of words and labels too
y_pred = []
x_pred = np.zeros((dp_n, ind_n))  # will be bag of words
dp_ind = 0
for label_dir in os.listdir(pred_dir):
    f = open(pred_dir + '/' + label_dir, "r")
    for line in f.readlines():
        for word in line.split():
            x_pred[dp_ind][word2ind[word]] += 1
        dp_ind += 1
        y_pred.append(int(label_dir[:-4]))
    f.close()
y_pred = np.array(y_pred)
print(x_pred.shape)
print(y_pred.shape)

# turn every dp to data frames
feat_cols = [i for i in range(ind_n)]
df = pd.DataFrame(x_pred, columns=feat_cols)
df['y'] = y_pred
df['label'] = df['y'].apply(lambda i: str(i))
print('Size of the dataframe: {}'.format(df.shape))

# randomizing the data (this might be needed, don't touch :))
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

### tsne
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
N = 300  # dp_n
df_subset = df.loc[rndperm[:N], :]
data_subset = df_subset[feat_cols].values
tsne_results = tsne.fit_transform(data_subset)

# plot
df_subset['tsne-2d-one'] = tsne_results[:, 0]
df_subset['tsne-2d-two'] = tsne_results[:, 1]
plt.figure(figsize=(16, 10))
scatter_plot = sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", df_subset['y'].nunique()),
    data=df_subset,
    legend="full",
    alpha=0.4
)
scatter_plot.legend(loc='center left', bbox_to_anchor=(2.0, 2.0))


plt.show()
fig = scatter_plot.get_figure()
fig.savefig("output.png")