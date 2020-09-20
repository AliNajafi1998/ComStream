from sklearn.manifold import TSNE
import os
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Params:
class TSNEPlot:
    def __init__(self, pred_dir, n_dp_to_plot, top_n, dist_outlier, word2ind_threshold, big_cluster_indexes2plot=None):
        self.top_n = top_n  # the clusters with dp's more than this number are valid, the rest are outliers
        self.dist_outlier = dist_outlier  # more means you will see more dp, more outliers
        self.n_dp_to_plot = n_dp_to_plot
        self.pred_dir = pred_dir
        self.word2ind = {}
        self.ind2word = {}
        self.ind_n = 0  # keep track of the number of new words in word2ind
        self.data_df = -1
        self.feat_cols = -1
        self.tsne_results = -1
        self.has_manulally_chosen_big_clusers = False
        self.big_cluster_indexes = []
        if big_cluster_indexes2plot is not None:
            self.big_cluster_indexes = big_cluster_indexes2plot
            self.has_manulally_chosen_big_clusers = True

        self.interval_indexes_in_data = []  # a list of indexes(end index -self not included)(the start is the end of the last
        self.word2ind_threshold = word2ind_threshold

    def fill_word2ind(self):
        word2ind2, ind2word2 = {}, {}
        good_inds = {}
        ind_n2 = 0
        for output_interval in os.listdir(self.pred_dir):
            for label_dir in os.listdir(self.pred_dir + '/' + output_interval + '/clusters'):
                f = open(self.pred_dir + '/' + output_interval + '/clusters/' + label_dir, "r")  # , encoding='utf8')
                for line in f.readlines():
                    for word in line.split():
                        if word not in word2ind2:
                            word2ind2[word] = ind_n2
                            ind2word2[ind_n2] = word
                            ind_n2 += 1
                        good_inds[word] = good_inds.get(word, 0) + 1
                f.close()

        for word, ind in word2ind2.items():
            if good_inds[word] >= self.word2ind_threshold:
                if word not in self.word2ind:
                    self.word2ind[word] = self.ind_n
                    self.ind2word[self.ind_n] = word
                    self.ind_n += 1
        print(f'old word2ind2 size: {len(word2ind2)}, new word2ind2 size: {len(self.word2ind)}')

    def fill_data_frame(self):
        # getting the real data to bag of words and labels too
        x_pred = []  # will be: list of bag of words
        y_pred = []  # a list of labels
        dp_ind = 0

        for output_interval in os.listdir(self.pred_dir):
            for label_dir in os.listdir(self.pred_dir + '/' + output_interval + '/clusters'):
                if self.has_manulally_chosen_big_clusers and int(
                        label_dir[:-4]) not in self.big_cluster_indexes:
                    continue
                f = open(self.pred_dir + '/' + output_interval + '/clusters/' + label_dir, "r")  # , encoding='utf8')
                for line in f.readlines():
                    line = line.strip()
                    if line == '':
                        continue
                    x_pred_one = np.zeros(self.ind_n)
                    for word in line.split():
                        if word in self.word2ind:
                            x_pred_one[self.word2ind[word]] += 1
                    if np.array_equal(x_pred_one, np.zeros(self.ind_n)):
                        continue
                    dp_ind += 1
                    y_pred.append(int(label_dir[:-4]))
                    x_pred.append(x_pred_one)
                f.close()
            self.interval_indexes_in_data.append(dp_ind)

        x_pred = np.array(x_pred)
        y_pred = np.array(y_pred)
        print(x_pred.shape)
        print(y_pred.shape)

        # turn every dp to data frames
        self.feat_cols = [i for i in range(self.ind_n)]
        df = pd.DataFrame(x_pred, columns=self.feat_cols)
        df['y'] = y_pred
        df['label'] = df['y'].apply(lambda i: str(i))
        self.data_df = df
        print('Size of the dataframe: {}'.format(self.data_df.shape))

    def run_tsne(self):
        self.data_df: pd.DataFrame
        # randomizing the data (this might be needed, don't touch :))
        # np.random.seed(42)
        # rndperm = np.random.permutation(self.data_df.shape[0])
        # print(f"number of big clusters: {self.data_df['y'].nunique()}")

        # tsne
        tsne_sklearn = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        # N: number of datapoints to be ploted 'len(self.data_df['y'])' means plot all of them)
        # N = min(self.n_dp_to_plot, len(self.data_df['y']))
        # self.df_subset = self.data_df.iloc[rndperm[:N], :]
        # data_subset = self.df_subset[self.feat_cols].values
        self.tsne_results = tsne_sklearn.fit_transform(self.data_df)
        self.data_df['tsne-2d-one'] = self.tsne_results[:, 0]
        self.data_df['tsne-2d-two'] = self.tsne_results[:, 1]
        print('tsne done!')

    def dist(self, x1, y1, x2, y2):
        return np.math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

    def outlier_removal(self):
        self.tsne_results: pd.DataFrame
        # get rid of outliers in only the plot:
        y_pred = self.data_df['y'].to_list()
        dist_ind = []  # tuple: (mean_distance, ind)
        N = len(self.data_df['y'])
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
            if count != 0:
                dist_ind.append((sum_dists / count, i))

        dist_ind.sort()
        non_outlier_indexes = []
        for i in range(len(dist_ind)):

            if i <= (len(dist_ind) - 1) / 3 or dist_ind[i - 1][0] > dist_ind[i][0] * self.dist_outlier:
                if y_pred[dist_ind[i][1]] in self.big_cluster_indexes:
                    non_outlier_indexes.append(dist_ind[i][1])
                else:
                    pass
            else:
                break
        # find the new: self.interval_indexes_in_data
        non_outlier_indexes.sort()
        new_interval_indexes_in_data = []
        now_old_interval_ind = 0
        for ind in range(len(non_outlier_indexes)):
            if non_outlier_indexes[ind] >= self.interval_indexes_in_data[now_old_interval_ind]:
                new_interval_indexes_in_data.append(ind)
                now_old_interval_ind += 1
        if len(new_interval_indexes_in_data) != len(self.interval_indexes_in_data):
            new_interval_indexes_in_data.append(len(non_outlier_indexes))
        print(f'old interval indexes: {self.interval_indexes_in_data}, new: {new_interval_indexes_in_data}')
        self.interval_indexes_in_data = new_interval_indexes_in_data
        print(f'Number of data points before getting rid of outliers: {self.data_df.shape}')
        self.data_df = self.data_df.iloc[non_outlier_indexes[:], :]
        print(f'Number of data points after getting rid of outliers: {self.data_df.shape}')

    # def make_outlier_special_color(self):
    #     y_pred = self.df_subset['y'].to_list()
    #     new_ys = []
    #     N = len(self.df_subset['y'])
    #     for i in range(N):
    #         if y_pred[i] not in self.big_cluster_indexes:
    #             new_ys.append(-1)
    #         else:
    #             new_ys.append(y_pred[i])
    #
    #     self.df_subset['y'] = new_ys

    def rearrange_label_names(self):
        self.data_df: pd.DataFrame
        count = 0
        label_names = sorted(self.data_df['y'].unique())
        new2old_label = {old_label: new_label for new_label, old_label in enumerate(label_names)}
        labels = self.data_df['y'].values
        for ind, old in enumerate(labels):
            labels[ind] = new2old_label[old] + 1
        self.data_df['y'] = labels

    def plot_tsne_results(self):
        self.data_df: pd.DataFrame
        # sns.palplot(sns.color_palette("Paired"))
        # flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        start_ind = 0
        for plot_ind, end_ind in enumerate(self.interval_indexes_in_data):
            df_subset = self.data_df.iloc[start_ind:end_ind]
            start_ind = end_ind
            plt.figure(figsize=(16, 16))
            scatter_plot = sns.scatterplot(
                x="tsne-2d-one", y="tsne-2d-two",
                hue="y",
                # palette=sns.color_palette("hls", df_subset['y'].nunique()),
                palette=sns.color_palette("hls", df_subset['y'].nunique()),
                data=df_subset,
                legend="full",
                alpha=0.4
                # sns.color_palette("Paired")
            )
            scatter_plot.legend(loc='center left', bbox_to_anchor=(0.0, 0.80))
            plt.show()
            fig = scatter_plot.get_figure()
            fig.savefig(str(plot_ind) + ".png")

    def run(self):
        self.fill_word2ind()
        self.fill_data_frame()
        self.run_tsne()
        self.outlier_removal()
        # self.make_outlier_special_color()
        self.rearrange_label_names()
        self.plot_tsne_results()


if __name__ == '__main__':
    tsne = TSNEPlot(pred_dir=str(Path(os.getcwd()).parent) + '/Algo/tmp_output',
                    # a file filled with text files filled with tweets, each text file is a cluster
                    n_dp_to_plot=10000,  # if you want to plot all, put 1e9 here
                    top_n=6,  # we will show the top_n clusters
                    dist_outlier=0.90,
                    word2ind_threshold=3,  # ignore the tfs less than this in the global dict
                    big_cluster_indexes2plot=[48, 34, 39, 47, 37,
                                              41])  # more means you will see more dp and outliers, 1.0 to see all
    tsne.run()
