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

# Params:
dp_threshold = 60  # the clusters with dp's more than this number are valid, the rest are outliers
visualization_outlier_threshold = 45  # more means you will see more dp, more outliers

path = Path(os.getcwd())

# assign the data dirs
true_dir = str(path.parent) + '/Data/reuters_cleaned.pkl'
pred_dir = str(path.parent) + '/Algo/output/'

# read the true and pred data
word2ind = {}
ind2word = {}
ind_n = 0  # keep track of the number of new words in word2ind
# fill the dict word2ind
for label_dir in os.listdir(pred_dir):
    f = open(pred_dir + '/' + label_dir, "r", encoding='utf8')
    for line in f.readlines():
        for word in line.split():
            if word not in word2ind:
                word2ind[word] = ind_n
                ind2word[ind_n] = word
                ind_n += 1
    f.close()

# getting the real data to bag of words and labels too
y_pred = []
x_pred = []  # will be bag of words
dp_ind = 0
for label_dir in os.listdir(pred_dir):
    f = open(pred_dir + '/' + label_dir, "r", encoding='utf8')
    if len(f.readlines()) < dp_threshold:
        continue
    f.close()
    f = open(pred_dir + '/' + label_dir, "r", encoding='utf8')
    for line in f.readlines():
        line = line.strip()
        x_pred_one = np.zeros(ind_n)
        for word in line.split():
            x_pred_one[word2ind[word]] += 1
        dp_ind += 1
        y_pred.append(int(label_dir[:-4]))
        x_pred.append(x_pred_one)
    f.close()

x_pred = np.array(x_pred)
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
print(f"number of big clusters: {df['y'].nunique()}")

# tsne
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
N = len(df['y'])  # number of datapoints to be ploted ('len(df['y'])' means plot all of them)
df_subset = df.loc[rndperm[:N], :]

# df_subset = df.iloc[:N, :]
data_subset = df_subset[feat_cols].values
tsne_results = tsne.fit_transform(data_subset)
df_subset['tsne-2d-one'] = tsne_results[:, 0]
df_subset['tsne-2d-two'] = tsne_results[:, 1]


# print(df_subset)

def dist(x1, y1, x2, y2):
    return np.math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))


# get rid of outliers in only the plot:
y_pred = df_subset['y'].to_list()
non_outlier_indexes = []
r = []
for i in range(N):
    sum_dists = 0
    count = 0
    for j in range(N):
        if y_pred[i] == y_pred[j] and i != j:
            distance = dist(tsne_results[i][0],
                            tsne_results[i][1],
                            tsne_results[j][0],
                            tsne_results[j][1])
            sum_dists += distance
            count += 1
    if sum_dists / count <= visualization_outlier_threshold:
        non_outlier_indexes.append(i)
    r.append(sum_dists / count)
df_subset = df_subset.iloc[non_outlier_indexes[:], :]
print(len(non_outlier_indexes))
print(sorted(r, reverse=True))
# tsne_results = np.array(new_tsne_results)
# print(df_subset)
N = len(df_subset['y'])  # number of datapoints to be ploted ('len(df['y'])' means plot all of them)

# if df_subset
# new_df_subset = new_df_subset.append(df_subset.iloc[x])
# words = ''
# for k in range(len(x_pred[max_dist_i])):
#     if x_pred[max_dist_i][k] != 0:
#         words += ind2word[k] + ' '
# debug
# print(f'label: {max_dist_label} distance: {max_dist_val} dp1:({tsne_results[max_dist_i][0],
# tsne_results[max_dist_i][1]}) dp2: ({tsne_results[max_dist_j][0], tsne_results[max_dist_j][1]})
# ')
# words = ''
# words2 = ''
# for k in range(len(x_pred[max_dist_i])):
#     if x_pred[max_dist_i][k] != 0:
#         words += ind2word[k] + ' '
#     if x_pred[max_dist_j][k] != 0:
#         words2 += ind2word[k] + ' '
# print(words)
# print(words2, end='\n\n\n')

# plot
# df_subset['tsne-2d-one'] = tsne_results[:, 0]
# df_subset['tsne-2d-two'] = tsne_results[:, 1]
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
