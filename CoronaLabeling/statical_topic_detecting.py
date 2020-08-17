import os
import numpy as np
from pathlib import Path
import pandas as pd

dir = os.path.join(Path(os.getcwd()).parent, 'Data/data_cleaned.pkl')
df = pd.read_pickle(dir)
df = df.iloc[np.random.RandomState(seed=42).permutation(len(df))[:1000000]]
df = df.sort_values(by=['created_at'])
df = df.reset_index().drop(['index'], axis=1)
# print(df.head())
df.to_pickle(os.path.join(Path(os.getcwd()).parent, 'Data/data_cleaned_1000k.pkl'))

# df['created_at'] = df['created_at'].apply(lambda x: x[:10])
# days = df['created_at'].unique()  # a list of ['2020-03-29' '2020-03-30'....]
# print(f'number of days: {len(days)}')

# # PARAMSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
# day_to_find = 1
# word_to_find = 'today'
# words_to_find = []
# ############################################ social country lockdown life government family quarantine distancing
# for day_to_find in range(33):
#     day_count = 0
#     for day in days:
#         if day_count < day_to_find:
#             day_count += 1
#             continue
#         print(f'the day is: {day}')
#         day_dict = {}
#         df_day = df.loc[df['created_at'] == day]
#         for ind, row in df_day.iterrows():
#             text = row['text']
#             for word in text.split():
#                 day_dict[word] = day_dict.get(word, 0) + 1
#         # most freq words each = {}
#         words_to_find = dict(
#             [(word, {}) for ind, word in sorted([(i, j) for j, i in day_dict.items()], reverse=True)[:100]])
#         print(f'most frequent terms: {sorted([(i, j) for j, i in day_dict.items()], reverse=True)[:100]}', end='\n\n\n')
#         break
#     news_with_the_word = []
#     day_count = 0
#     for day in days:
#         if day_count < day_to_find:
#             day_count += 1
#             continue
#
#         day_dict = {}
#         df_day = df.loc[df['created_at'] == day]
#         for ind, row in df_day.iterrows():
#             text = row['text']
#             for word_to_find in words_to_find:
#                 if word_to_find in text.split():
#                     for word in text.split():
#                         words_to_find[word_to_find][word] = words_to_find[word_to_find].get(word, 0) + 1
#         break
#     file_dir = str(os.getcwd()) + '/days/'
#     if not os.path.exists(file_dir):
#         os.makedirs(file_dir)
#     f = open(file_dir + str(day) + '.txt', "w")
#     for main_hot_word, dics in words_to_find.items():
#         print(sorted([(i, j) for j, i in dics.items()], reverse=True)[:30], end='\n\n')
#         string = ''
#         for freq, word in sorted([(i, j) for j, i in dics.items()], reverse=True)[:30]:
#             string+=word
#             string+=' '
#         string+='\n\n'
#         f.write(string)
#     f.close()
#
# # for news in news_with_the_word:
# #     print(news)
