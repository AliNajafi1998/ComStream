from sklearn.manifold import TSNE
import os
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Params:
class TsnePlot():
    def __init__(self, pred_dir, n_dp_to_plot, dp_threshold, visualization_outlier_threshold):
        self.dp_threshold = dp_threshold  # the clusters with dp's more than this number are valid, the rest are outliers
        self.visualization_outlier_threshold = visualization_outlier_threshold  # more means you will see more dp, more outliers
        self.n_dp_to_plot = n_dp_to_plot
        self.pred_dir = pred_dir
        self.word2ind = {}
        self.ind2word = {}
        self.ind_n = 0  # keep track of the number of new words in word2ind
        self.data_df = -1
        self.feat_cols = -1
        self.df_subset = -1
        self.tsne_results = -1

    def fill_word2ind(self):
        for label_dir in os.listdir(self.pred_dir):
            f = open(self.pred_dir + '/' + label_dir, "r", encoding='utf8')
            for line in f.readlines():
                for word in line.split():
                    if word not in self.word2ind:
                        self.word2ind[word] = self.ind_n
                        self.ind2word[self.ind_n] = word
                        self.ind_n += 1
            f.close()

    def fill_data_frame(self):
        # getting the real data to bag of words and labels too
        x_pred = []  # will be list of bag of words
        y_pred = []
        dp_ind = 0
        for label_dir in os.listdir(self.pred_dir):
            f = open(self.pred_dir + '/' + label_dir, "r", encoding='utf8')
            if len(f.readlines()) < self.dp_threshold:
                continue
            f.close()
            f = open(self.pred_dir + '/' + label_dir, "r", encoding='utf8')
            for line in f.readlines():
                line = line.strip()
                x_pred_one = np.zeros(self.ind_n)
                for word in line.split():
                    x_pred_one[self.word2ind[word]] += 1
                dp_ind += 1
                y_pred.append(int(label_dir[:-4]))
                x_pred.append(x_pred_one)
            f.close()

        x_pred = np.array(x_pred)
        y_pred = np.array(y_pred)
        print(x_pred.shape)
        print(y_pred.shape)

        # turn every dp to data frames
        self.feat_cols = [i for i in range(self.ind_n)]
        self.data_df = pd.DataFrame(x_pred, columns=self.feat_cols)
        self.data_df['y'] = y_pred
        self.data_df['label'] = self.data_df['y'].apply(lambda i: str(i))
        print('Size of the dataframe: {}'.format(self.data_df.shape))

    def run_tsne(self):
        # randomizing the data (this might be needed, don't touch :))
        np.random.seed(42)
        rndperm = np.random.permutation(self.data_df.shape[0])
        print(f"number of big clusters: {self.data_df['y'].nunique()}")
        # tsne
        tsne_sklearn = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        # N: number of datapoints to be ploted 'len(self.data_df['y'])' means plot all of them)
        N = min(self.n_dp_to_plot, len(self.data_df['y']))
        self.df_subset = self.data_df.loc[rndperm[:N], :]  # should it be iloc??????
        data_subset = self.df_subset[self.feat_cols].values
        self.tsne_results = tsne_sklearn.fit_transform(data_subset)
        self.df_subset['tsne-2d-one'] = self.tsne_results[:, 0]
        self.df_subset['tsne-2d-two'] = self.tsne_results[:, 1]

    def dist(self, x1, y1, x2, y2):
        return np.math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

    def outlier_removal(self):
        # get rid of outliers in only the plot:
        y_pred = self.df_subset['y'].to_list()
        non_outlier_indexes = []
        N = len(self.df_subset['y'])
        for i in range(N):
            sum_dists = 0
            count = 0
            for j in range(N):
                if y_pred[i] == y_pred[j] and i != j:
                    distance = self.dist(self.tsne_results[i][0],
                                         self.tsne_results[i][1],
                                         self.tsne_results[j][0],
                                         self.tsne_results[j][1])
                    sum_dists += distance
                    count += 1
            if count != 0 and sum_dists / count <= self.visualization_outlier_threshold:
                non_outlier_indexes.append(i)
        self.df_subset = self.df_subset.iloc[non_outlier_indexes[:], :]
        print(f'Number of data points after getting rid of outliers: {len(non_outlier_indexes)}')

    def plot_tsne_results(self):
        plt.figure(figsize=(16, 10))
        scatter_plot = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="y",
            palette=sns.color_palette("hls", self.df_subset['y'].nunique()),
            data=self.df_subset,
            legend="full",
            alpha=0.4
        )
        scatter_plot.legend(loc='center left', bbox_to_anchor=(2.0, 2.0))
        plt.show()
        fig = scatter_plot.get_figure()
        fig.savefig("output.png")

    def run(self):
        self.fill_word2ind()
        self.fill_data_frame()
        self.run_tsne()
        self.outlier_removal()
        self.plot_tsne_results()


if __name__ == '__main__':
    tsne = TsnePlot(pred_dir=str(Path(os.getcwd()).parent) + '/Algo/output/',
                    n_dp_to_plot=5000,  # if you want to plot all, put 1e9 here
                    dp_threshold=20,  # the clusters with dp's more than this number are valid, the rest are outliers
                    visualization_outlier_threshold=70)  # more means you will see more dp more outliers, put 1e9 to see all
    tsne.run()
