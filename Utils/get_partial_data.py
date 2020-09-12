import os
import numpy as np
from pathlib import Path
import pandas as pd

dir = os.path.join(Path(os.getcwd()).parent, 'Data/data_cleaned.pkl')
df = pd.read_pickle(dir)
print(len(df))
df = df.iloc[np.random.RandomState(seed=42).permutation(len(df))[:1000000]]
df = df.sort_values(by=['created_at'])
df = df.reset_index().drop(['index'], axis=1)
df.to_pickle(os.path.join(Path(os.getcwd()).parent, 'Data/data_cleaned_1000k.pkl'))
