"""
    VQA2.0 dataset class
"""
import os
import pickle
from PIL import Image

import numpy as np
from random import randint
import torch.utils.data as data
from Utils.util import pad_sentence


def default_loader(path):
    return Image.open(path).convert('RGB')


class WordMatchDataset(data.Dataset):
    def __init__(self, opt, data_file, loader=default_loader):

        self.opt = opt
        self.loader = loader
        self.max_q_len, self.max_c_len = opt.q_max_sentence_len, opt.c_max_sentence_len
        self.img_dir = opt.img_dir

        self.f = pickle.load(open(data_file, "rb"))
        self.data = self.f["data"]
        # vocabulary dictionaries
        self.c_i2w, self.c_w2i = self.f["c_dicts"]
        self.p_i2w, self.p_w2i = self.f["pos_dicts"]
        self.q_i2w, self.q_w2i = self.f["q_dicts"]
        self.a_i2w, self.a_w2i = self.f["a_dicts"]
        self.c_vocab_size = len(self.c_i2w)
        self.p_vocab_size = len(self.p_i2w)
        self.q_vocab_size = len(self.q_i2w)
        self.a_vocab_size = len(self.a_i2w)
        self.special_symbols = self.f["special_symbols"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        sample = self.data[index]

        # prepare question
        question = sample['question'][:self.max_q_len]
        question_len = len(question)+1

        source = [self.special_symbols['bos']] + question  # add <bos> token
        target = question + [self.special_symbols['eos']]  # add <eos> token
        source = pad_sentence(source, max_len=self.max_q_len+1, pad_idx=self.q_vocab_size)
        target = pad_sentence(target, max_len=self.max_q_len+1, pad_idx=self.q_vocab_size)

        answer = sample['answer']

        # get image
        img_path = os.path.join(self.img_dir, "features", str(sample['image_id']) + '.npy')
        img = np.load(img_path)
        img_file = os.path.join(self.img_dir, sample['img_raw_folder'], sample['img_raw_file'])

        # prepare pos and caption
        # pos = sample['pos']
        caption = [self.special_symbols['bos']] + sample['caption']
        caption = caption[:self.max_c_len]  # also limit the length of captions
        caption = pad_sentence(caption, max_len=self.max_c_len+1, pad_idx=self.c_vocab_size)

        refs = []
        ref_lens = []
        for ref in sample['refs']:
            cap = ref['caption'][:self.max_c_len]
            refs.append(pad_sentence(cap, max_len=self.max_c_len, pad_idx=self.c_vocab_size))
            ref_lens.append(len(cap))

        q_idx = sample['q_idx']
        q_idx_vec = np.zeros((self.max_c_len, self.opt.cap_rnn_size))
        q_idx_vec[q_idx, :] = 1.0

        return img, question_len, np.asarray(source), np.asarray(target), np.asarray(caption), q_idx, q_idx_vec, np.asarray(refs), answer, img_file

    # def get_item_with_words(self, index):
    #     sample = self.data[index]
    #
    #     img_path = os.path.join(self.opt.coco_dir + '/images', sample['img_raw_folder'], sample['img_raw_file'])
    #     img = self.loader(img_path)
    #
    #     question = sample['question']
    #     q_words = []
    #     for ind in question:
    #         q_words.append(self.q_i2w[ind])
    #
    #     img_feat, _, _, _, caption, q_idx, q_idx_vec, captions, answer = self.__getitem__(index)
    #
    #     return img, q_words, caption, img_feat, q_idx, q_idx_vec, captions, self.c_i2w[answer]
    #
    # def get_items(self, idx_arr):
    #     img, q_words, caption, img_feat, q_idx, q_idx_vec, captions, answer = [], [], [], [], [], [], [], []
    #     for i in idx_arr:
    #         for l, item in zip([img, q_words, caption, img_feat, q_idx, q_idx_vec, captions, answer], self.get_item_with_words(i)):
    #             l.append(item)
    #
    #     return img, q_words, caption, img_feat, q_idx, q_idx_vec, captions, answer


class PosMatchDataset(WordMatchDataset):
    def __init__(self, opt, data_file, fake_len, loader=default_loader):

        super(PosMatchDataset, self).__init__(opt, data_file, loader=loader)

        self.real_len = len(self.data)
        self.fake_len = fake_len

    def __len__(self):
        return self.fake_len

    def __getitem__(self, index):

        if index >= self.real_len:
            i = randint(0, self.real_len-1)
            return super(PosMatchDataset, self).__getitem__(i)
        else:
            return super(PosMatchDataset, self).__getitem__(index)
