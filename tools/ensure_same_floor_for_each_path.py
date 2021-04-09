import pandas as pd
from datetime import datetime
import os

base_path = '..'

submission_path = f'{base_path}/input/submissions/submission_lstm_floor_pred.csv'

subm = pd.read_csv(submission_path, index_col=0)
test_data = pd.read_csv(submission_path)

test_data['path'] = test_data['site_path_timestamp'].str.split(pat='_', n=- 1, expand=True)[1]
test_data['p_floor'] = test_data['floor'].astype(int)


# re-elaboration taking the median
def modee1(a):
    return (a['path'].unique())


def modee2(a):
    return (a['p_floor'].value_counts().head(1).reset_index()['index'].values[0])


dft = pd.DataFrame()
# df['path']=X_ass_val.groupby('path').apply(modee1)
dft['my_b_floor_pred'] = test_data.groupby('path').apply(modee2)
test_data = test_data.merge(dft, how='left', on='path')
test_data.index = subm.index
subm['floor'] = test_data['my_b_floor_pred']

# Current date and time
now = datetime.now()
timestamp = datetime.timestamp(now)
subm.to_csv(f'submission_ensure_same_floor_for_each_path.csv')
