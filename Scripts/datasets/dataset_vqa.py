"""
    VQA2.0 dataset class
"""
import os
import pickle
from random import randint
from PIL import Image

import numpy as np
import torch.utils.data as data
from Utils.util import pad_sentence


def default_loader(path):
    return Image.open(path).convert('RGB')


class VQADataset(data.Dataset):
    def __init__(self, opt, data_file, loader=default_loader):

        self.img_dir = opt.img_dir
        self.loader = loader
        self.multi_answer = opt.multi_answer
        self.max_c_len = opt.c_max_sentence_len
        self.max_q_len = opt.q_max_sentence_len

        self.f = pickle.load(open(data_file, "rb"))
        self.data = self.f["data"]
        # vocabulary dictionaries
        self.c_i2w, self.c_w2i = self.f["c_dicts"]
        self.q_i2w, self.q_w2i = self.f["q_dicts"]
        self.a_i2w, self.a_w2i = self.f["a_dicts"]
        self.c_vocab_size = len(self.c_i2w)
        self.q_vocab_size = len(self.q_i2w)
        self.a_vocab_size = len(self.a_i2w)
        self.special_symbols = self.f["special_symbols"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        sample = self.data[index]

        # get question
        question = sample['question'][:self.max_q_len]
        question_len = len(question)

        question = pad_sentence(question, max_len=self.max_q_len, pad_idx=self.q_vocab_size)

        # get image
        img_path = os.path.join(self.img_dir, "features", str(sample['image_id']) + '.npy')
        img = np.load(img_path)
        img_file = os.path.join(self.img_dir, sample['img_raw_folder'], sample['img_raw_file'])

        # get answers
        answers = sample['answers']

        if self.multi_answer:
            label = np.zeros(self.a_vocab_size)
            for word, confidence in answers:
                label[word] = float(confidence)
        else:
            if len(answers) > 0:
                label = answers[0][0]
            else:
                label = randint(0, self.a_vocab_size-1)   # set it to random word if there are no answers (hack)

        # get reference captions
        refs = []
        ref_lens = []
        for ref in sample['refs']:
            cap = ref['caption'][:self.max_c_len]
            refs.append(pad_sentence(cap, max_len=self.max_c_len, pad_idx=self.c_vocab_size))
            ref_lens.append(len(cap))

        return img, np.asarray(question), question_len, label.astype(np.float32), np.asarray(refs), np.asarray(ref_lens), img_file, sample['question_id']