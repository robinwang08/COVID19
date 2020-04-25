import pandas as pd
import numpy as np
import glob
import os, os.path as osp
import re
from tqdm import tqdm

#Generate support files to analyze all slices in CT

ROOT = '/home/user1/4TBHD/COR19_without_seg'

numpy_arrays = glob.glob(osp.join(ROOT, 'all_slices/*.npy'))
# Ignore the ones in fixed_nrrd, I already replaced the broken ones
numpy_arrays = [_ for _ in numpy_arrays if not re.search('fixed_nrrd', _)]

labels_df = pd.read_csv(osp.join(ROOT, 'all_815_slices.csv'))
labels_df = labels_df.drop_duplicates()
labels_df['Unnamed: 2'].iloc[0] = '9-25'
labels_df = labels_df[~labels_df['Slices'].isna()]
labels_df = labels_df.drop([357, 354, 635]).reset_index(drop=True)
labels_dict = {} ; slice_cols = ['Slices'] + ['Unnamed: {}'.format(_) for _ in range(2,8)]
for i, _df in labels_df.groupby('AICode'):
    assert len(_df) == 1
    _slices = np.asarray(_df[slice_cols])[0]
    _slices = [_ for _ in _slices if str(_) != 'nan']
    slice_ranges = []
    for _ in _slices:
        slice_range = _.split('-')
        if len(slice_range) == 1:
            slice_range = int(slice_range[0])
        else:
            assert len(slice_range) == 2
            slice_range = tuple([int(_) for _ in slice_range])
        slice_ranges.append(slice_range)
    positive_slices = []
    for _ in slice_ranges:
        if type(_) == int:
            positive_slices.extend([_])
        elif type(_) == tuple:
            positive_slices.extend(list(range(_[0], _[1]+1)))
    positive_slices = np.asarray(positive_slices) - 1 # 1-indexed so subtract 1
    labels_dict[i] = np.unique(np.sort(positive_slices))


slice_df = pd.DataFrame({
        'filename': numpy_arrays,
        'AICode': [_.split('/')[-1].split('-')[0].upper() for _ in numpy_arrays],
        'slice_num': [int(_.split('/')[-1].split('-')[-1].replace('.npy', '')) for _ in numpy_arrays]
    })

slice_df['group'] = [re.sub(r'[0-9]+', '', _) for _ in slice_df['AICode']]
slice_df['group'] = [_.replace('.', '') for _ in slice_df['group']]

# Check for overlap
list(set(slice_df['AICode']) - set(labels_df['AICode']))
list(set(labels_df['AICode']) - set(slice_df['AICode']))

slice_df = slice_df.merge(labels_df, on='AICode')
slice_df = slice_df[~slice_df['Slices'].isna()]
slice_df['split'] = 'train'


'''
test_df = pd.read_csv(osp.join(ROOT, 'test_set.csv'))
test_df['AICode'] = [_.split('/')[-1].split('-')[0].upper() for _ in test_df['nrrdfile']]

slice_df = slice_df[~slice_df['Slices'].isna()]
slice_df['split'] = 'train'
slice_df.loc[slice_df['AICode'].isin(test_df['AICode']),'split'] = 'test'
'''

slice_df['label'] = 0
for rownum in tqdm(range(len(slice_df)), total=len(slice_df)):
    slice_number = slice_df.iloc[rownum]['slice_num']
    ai_code = slice_df.iloc[rownum]['AICode']
    if slice_number in labels_dict[ai_code]:
        if re.search('COR', ai_code):
            slice_df['label'].iloc[rownum] = 2
        else:
            slice_df['label'].iloc[rownum] = 1

'''
np.random.seed(88)
valid_ids = np.random.choice(slice_df['AICode'].unique(), int(0.1*len(slice_df['AICode'].unique())), replace=False)
slice_df.loc[slice_df['AICode'].isin(valid_ids), 'split'] = 'valid'
'''

slice_df = slice_df.sort_values(['AICode', 'slice_num'])
slice_df.to_csv(osp.join(ROOT, 'train_with_splits.csv'), index=False)

# Examine 1,000 positive slices at random
'''
np.random.seed(88)
pos_slices = np.random.choice(slice_df['filename'][slice_df['label'] == 1], 1000, replace=False)

SAVEDIR = osp.join(ROOT, 'check_pos_arrays')
if not os.path.exists(SAVEDIR): os.makedirs(SAVEDIR)

import cv2

def window(x, wl, ww):
    upper = wl + ww / 2
    lower = wl - ww / 2
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x)
    x = x * 255.
    return x.astype('uint8')

for _ in pos_slices:
    X = np.load(_)
    X = window(X, -600, 1500)
    X = np.expand_dims(X, axis=-1)
    X = cv2.cvtColor(X, cv2.COLOR_GRAY2RGB)
    savefile = '_'.join(_.split('/')[-2:]).replace('npy', 'png')
    status = cv2.imwrite(osp.join(SAVEDIR, savefile), X)
'''


