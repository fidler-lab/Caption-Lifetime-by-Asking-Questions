import os
import pickle
from PIL import Image

import numpy as np
import torch.utils.data as data
from Utils.util import pad_sentence


def default_loader(path):
    return Image.open(path).convert('RGB')


class CapDataset(data.Dataset):
    def __init__(self, opt, data_file, loader=default_loader):

        self.img_dir = opt.img_dir
        self.loader = loader
        self.max_c_len = opt.c_max_sentence_len

        self.f = pickle.load(open(data_file, "rb"))
        self.data = self.f["data"]
        # vocabulary dictionaries
        self.c_i2w, self.c_w2i = self.f["c_dicts"]
        self.p_i2w, self.p_w2i = self.f["pos_dicts"]
        self.q_i2w, self.q_w2i = self.f["q_dicts"]
        self.a_i2w, self.a_w2i = self.f["a_dicts"]
        self.c_vocab_size = len(self.c_i2w)
        self.pos_vocab_size = len(self.p_i2w)
        self.q_vocab_size = len(self.q_i2w)
        self.a_vocab_size = len(self.a_i2w)
        self.special_symbols = self.f["special_symbols"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        sample = self.data[index]

        caption = sample['captions'][sample['cap_index']]
        caption = caption['caption'][:self.max_c_len]

        img, source, target, caption_len, refs, ref_lens, pos, img_path = self._get_content(caption, sample)

        return img, np.asarray(source), np.asarray(target), caption_len, np.asarray(refs), np.asarray(ref_lens),\
               np.asarray(pos), img_path, index, sample['image_id']

    def _get_content(self, caption, sample):
        # get caption
        source = [self.special_symbols['bos']] + caption  # add <bos> token
        target = caption + [self.special_symbols['eos']]  # add <eos> token
        source = pad_sentence(source, max_len=self.max_c_len+1, pad_idx=self.c_vocab_size)
        target = pad_sentence(target, max_len=self.max_c_len+1, pad_idx=self.c_vocab_size)

        # get image
        img_path = os.path.join(self.img_dir, "features", str(sample['image_id']) + '.npy')
        img = np.load(img_path)
        img_file = os.path.join(self.img_dir, sample['img_raw_folder'], sample['img_raw_file'])

        # get reference captions
        refs = []
        ref_lens = []
        for ref in sample['captions']:
            cap = ref['caption'][:self.max_c_len]
            refs.append(pad_sentence(cap, max_len=self.max_c_len + 1, pad_idx=self.c_vocab_size))
            ref_lens.append(len(cap))

        # get part-of-speech
        pos = sample['pos'][:self.max_c_len]
        assert len(pos) == len(caption)

        pad_len = max(0, self.max_c_len - len(pos)) + 1  # +1 for the <eos> token
        pos = pos + [self.pos_vocab_size] * pad_len

        return img, source, target, len(caption)+1, refs, ref_lens, pos, img_file