"""
Split a single caption training file into N splits. The first split is warmup. The other splits are used in lifelong
learning. For lifelong learning splits, there are num_caps captions per image.
"""

import pickle
import argparse
import random
import numpy as np

np.random.seed(1)
random.seed(1)

def remove_datasamples(data, split, num_keep=2):

    img2idx = {}
    for i, item in enumerate(data):
        imgId = item['image_id']
        if imgId in img2idx:
            img2idx[imgId].append(i)
        else:
            img2idx[imgId] = [i]

    # For lifelong splits, keep only num_keep captions for each image
    if split > 0:
        new_data = []
        for img, idxs in img2idx.items():
            random.shuffle(idxs)
            for i in range(len(idxs)):
                if i > num_keep-1:
                    break
                new_data.append(data[idxs[i]])  # keep only num_keep random samples
        print("There are {} images, {} captions in split {}.".format(len(img2idx), len(new_data), split))
        return new_data
    else:
        print("There are {} images, {} captions in split {} (warmup).".format(len(img2idx), len(data), split))
        return data


def split_full_data(params):
    dset = pickle.load(open(params.data_file, 'rb'))
    data = dset['data']

    first_idx = int(len(data)*(params.warmup/100.0))
    inc = (len(data)-first_idx)/(params.num_splits-1)

    # warmup split
    first_split = data[0:first_idx]
    # lifelong splits
    splits = [first_split]
    i = first_idx
    while i+inc < len(data):
        splits.append(data[i: i+inc])
        i += inc

    # save splits
    for i, split in enumerate(splits):
        split = remove_datasamples(split, i, num_keep=params.num_caps)
        dset['data'] = split
        pickle.dump(dset, open(params.output_file+"_{}.p".format(i), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--data_file', default='Data/annotation/cap_train.p', help='input caption training file')
  parser.add_argument('--output_file', default='Data/annotation/train_split', help='output lifelong splits')
  parser.add_argument('--warmup', type=float, default=10, help='percent of annotation in warmup')
  parser.add_argument('--num_splits', type=int, default=4, help='number of total annotation splits')
  parser.add_argument('--num_caps', type=int, default=2, help='number of captions per annotation split')
  params = parser.parse_args()

  split_full_data(params)
