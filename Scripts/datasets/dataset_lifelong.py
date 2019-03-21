from dataset_caption import CapDataset
from PIL import Image
import numpy as np
import pickle
import torch
import operator
from stanfordcorenlp import StanfordCoreNLP
from Utils.util import to_np, idx2str, clean_str

def default_loader(path):
    return Image.open(path).convert('RGB')


class LifelongDataset(CapDataset):
    def __init__(self, opt, data_file, loader=default_loader):

        super(LifelongDataset, self).__init__(opt, data_file, loader)
        self.gt_reward = self.f["gt_caps_reward"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        caption = sample['caption'][:self.max_c_len]

        img, source, target, caption_len, refs, ref_lens, pos, img_path = self._get_content(caption, sample)

        return img, np.asarray(source), np.asarray(target), caption_len, np.asarray(pos), sample['weight'].astype(np.float)


class DataCollector:

    def __init__(self, opt, data_file):


        self.img_dir = opt.img_dir

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

        self.model = opt.model
        self.max_c_len = opt.c_max_sentence_len
        self.stanfordnlp = StanfordCoreNLP(opt.stanfordnlp_dir)
        self.H, self.lamda = opt.H, opt.lamda

        self.data_dump = []
        self.epoch_collects, self.epoch_redundant_collects = 0, 0

        self.collected_data = []
        self.question_set = set()
        self.no_question_set = set()
        self.collection_stats = {}

        self.create_imgid2idx_map()
        self.init_collected_data()

    def create_imgid2idx_map(self):
        self.imgId2idx = {}

        for i, item in enumerate(self.data):
            imgId = item['image_id']
            if imgId in self.imgId2idx:
                self.imgId2idx[imgId].append(i)
            else:
                self.imgId2idx[imgId] = [i]

    def init_collected_data(self):
        """
        The captions are stored as dictionaries from tuple(caption) --> reward/score, this ensures uniqueness of the
        gathered captions since the keys of the dictionary can be treated as a set
        """
        for i in range(len(self.data)):

            item = self.data[i].copy()
            caption = item['captions'][item['cap_index']]
            item['caption'], item['caption_id'] = caption['caption'], caption['caption_id']

            self.collected_data.append({
                'max_reward': 0.0,
                'best_cap_type': -1,  # -1 for ground truth, 0 for gathered w/o question, 1 for gathered w/ question
                'gt_data': item,
                'cap_dict': {},  # keeps track of all the captions seen for this image, and its reward
                'ask_cap_dict': {},  # same thing but only captions where a question was asked
                'best_cap_dict': {},  # keep track of all the best captions between rollout, replace, original
                'best_ask_cap_dict': {}  # same thing but a captions where a question was asked
            })

    def collect_batch(self, index, captions, masks, rewards, ans=None, qs=None, qlens=None, topk=None):

        index, captions, lens = to_np(index), [to_np(x) for x in captions], [to_np(torch.sum(x, dim=1)) for x in masks]
        if self.model == "QuestionAskingTrainer":
            ans, qs, qlens, topk = [to_np(x) for x in ans], [to_np(x) for x in qs],\
                                   [to_np(x) for x in qlens], [to_np(x) for x in topk]
        else:
            ans, qs, qlens, topk = [], [], [], []

        self.data_dump.append([index, captions, lens, rewards, ans, qs, qlens, topk])

    def process_collected_data(self):

        for index, captions, lens, rewards, ans, qs, qlens, topk in self.data_dump:

            rewards = np.stack(rewards, axis=1)
            maxR_idx = np.argmax(rewards, axis=1)
            maxR = np.max(rewards, axis=1)

            # i iterates over this batch, idx is the idx for data in the dataset
            for i, idx in enumerate(index):
                Ri = maxR[i]
                best_cap_type = maxR_idx[i]
                entry_to_update = self.collected_data[idx]

                self.set_unique_captions(captions, lens, rewards, i, idx, best_cap_type)

                # set a new best caption
                if Ri > entry_to_update['max_reward']:

                    # update the best reward
                    entry_to_update['max_reward'] = Ri

                    # update some stats based on whether the best caption was from asking questions or not
                    old_type = entry_to_update['best_cap_type']
                    self.update_epoch_gather_stats(best_cap_type, old_type, idx)
                    entry_to_update['best_cap_type'] = best_cap_type

                    # update the question that was asked, the answer from the teacher, and the top-k words
                    if self.model == "QuestionAskingTrainer":

                        if best_cap_type > 0:
                            entry_to_update['answer'] = ans[best_cap_type-1][i]
                            entry_to_update['question'] = qs[best_cap_type-1][i][:qlens[best_cap_type-1][i]]
                            entry_to_update['topk'] = topk[best_cap_type-1][i]
                        else:
                            entry_to_update['answer'] = None
                            entry_to_update['question'] = None
                            entry_to_update['topk'] = None

        self.data_dump = []

    def set_unique_captions(self, captions, lens, rewards, i, idx, best_type):

        slp_data = self.collected_data[idx]

        for best_cap_type in range(3):

            caption = captions[best_cap_type][i]
            cap_len = lens[best_cap_type][i]
            stripped_caption = caption[:cap_len].tolist()
            reward = rewards[i][best_cap_type]
            slp_data['cap_dict'][tuple(stripped_caption)] = reward  # we're ok overwriting since reward is always same

            if best_cap_type == best_type: slp_data['best_cap_dict'][tuple(stripped_caption)] = reward

            if best_cap_type > 0:
                slp_data['ask_cap_dict'][tuple(stripped_caption)] = reward
                if best_cap_type == best_type: slp_data['best_ask_cap_dict'][tuple(stripped_caption)] = reward

    def get_average_reward_each_image(self):
        img_avg_rwd = []

        for imgId, idxs in self.imgId2idx.items():
            total_score = 0.0
            for idx in idxs:
                total_score += self.collected_data[idx]['max_reward']
            avg_score = total_score/len(idxs)
            img_avg_rwd.append((imgId, avg_score))

        img_avg_rwd.sort(key=lambda tup: tup[1])
        return img_avg_rwd

    def get_sample_best_caption(self, idx):

        entry = self.collected_data[idx]
        sample = entry['gt_data'].copy()

        # assign the best caption to the sample
        best_cap = sorted(entry['cap_dict'].items(), key=operator.itemgetter(1), reverse=True)[0]
        sample['caption'] = list(best_cap[0])
        sample['weight'] = entry['max_reward']

        assert best_cap[1] == entry['max_reward']

        # additional metrics
        if self.model == "QuestionAskingTrainer":
            sample['col_ans'] = entry['answer']
            sample['col_question'] = entry['question']
            sample['col_topk'] = entry['topk']

        return sample

    def clean_cap_pos(self, cap_str, pos_str):
        single_word_str = ' '.join(cap_str).split()

        new_cap = [self.c_w2i[x] if x in self.c_w2i else self.special_symbols['unknown_word'] for x in single_word_str]

        if '-LRB-' in pos_str:
            pos_str.remove('-LRB-')
        if '-RRB-' in pos_str:
            pos_str.remove('-RRB-')

        return new_cap, pos_str

    # At the end of every chunk, save the collected data. We keep the top H% percent of images + captions and give up
    # on 1-H% images (getting ground truth captions for them)
    def save_collected_data(self, output_dir, logger):
        img_avg_rwd = self.get_average_reward_each_image()

        imgs_to_gather = int(len(self.imgId2idx) * self.H)
        gt_reward_idx = min(int(len(img_avg_rwd) - (1 - self.lamda) * imgs_to_gather), len(img_avg_rwd) - 1)
        gt_reward = img_avg_rwd[gt_reward_idx][1]

        new_data = []
        imgs_collected, captions_collected, skipped_pos_parse_error = 0, 0, 0
        q_gathers, no_q_gathers, total_cider = 0, 0, 0.0
        gaveup_list = []

        img_avg_rwd.reverse()

        # collect the best caption for the top H% images
        for imgId, _ in img_avg_rwd:

            idxs = self.imgId2idx[imgId]

            if imgs_collected < imgs_to_gather:

                for idx in idxs:
                    item = self.get_sample_best_caption(idx)

                    caption = item['caption']
                    cap_str = idx2str(self.c_i2w, caption)
                    pos_str = [p for w, p in self.stanfordnlp.pos_tag(' '.join([clean_str(x) for x in cap_str]))]

                    cap_clean, pos_clean = self.clean_cap_pos(cap_str, pos_str)

                    item['caption'] = cap_clean
                    caption = item['caption']

                    if len(pos_clean) == len(caption):

                        if idx in self.question_set:

                            q_gathers += 1
                            item['caption_type'] = 1
                        elif idx in self.no_question_set:

                            no_q_gathers +=1
                            item['caption_type'] = 2

                        item['pos'] = [self.p_w2i[x] if x in self.p_w2i else self.p_w2i['unknown'] for x in pos_clean]

                        new_data.append(item)

                        captions_collected += 1
                        total_cider += item['weight']

                    else:

                        gaveup_list.append(idx)
                        skipped_pos_parse_error += 1

            else:
                for idx in idxs:
                    gaveup_list.append(idx)

            imgs_collected += 1

        # get the ground truth captions for the rest of the images
        for idx in gaveup_list:
            item = self.collected_data[idx]['gt_data']
            item['weight'] = gt_reward
            item['caption_type'] = 0
            new_data.append(item)

        assert len(new_data) == len(self.data)

        # Logging
        logger.info("Collecting the top {}/{} images.".format(imgs_to_gather, len(self.imgId2idx)))
        logger.info("Collected {}/{} captions, and asked {}/{} samples for ground truth.".format(
            captions_collected, len(self.data), len(gaveup_list), len(self.data)))
        logger.info("Of the {} captions collected, {} were from asking a question and {} were from not.".format(
            captions_collected, q_gathers, no_q_gathers))
        logger.info("{} captions were skipped due to pos parsing error".format(skipped_pos_parse_error))
        logger.info("The average gathered reward was {} and the 80 percentile reward (GT reward is set to this) is {}.".format(
            total_cider/captions_collected, gt_reward))

        # Save data
        pickle.dump({
            "data": new_data,
            "c_dicts": [self.c_i2w, self.c_w2i],
            "pos_dicts": [self.p_i2w, self.p_w2i],
            "a_dicts": [self.a_i2w, self.a_w2i],
            "q_dicts": [self.q_i2w, self.q_w2i],
            "special_symbols": self.special_symbols,
            "gt_caps_reward": gt_reward
        }, open(output_dir, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        stats = {k:v for k,v in zip(
            ['collects', 'num gt', 'q collects', 'no q collects', 'avg collect reward', 'gt reward', 'pos parse error'],
            [captions_collected, len(gaveup_list), q_gathers, no_q_gathers, total_cider/captions_collected, gt_reward, skipped_pos_parse_error])}
        self.set_gather_stats(stats)

    def get_epoch_stats(self):

        question_distr = []
        no_question_distr = []

        for idx in self.question_set:
            slp_data = self.collected_data[idx]
            question_distr.append(slp_data['max_reward'])

        for idx in self.no_question_set:
            slp_data = self.collected_data[idx]
            no_question_distr.append(slp_data['max_reward'])

        info = {k: v for k, v in zip(
            ["epoch captions collected", "epoch redundant collected",
             "best cap from asking question", "best cap from sampling"],
            [self.epoch_collects, self.epoch_redundant_collects, len(self.question_set),
             len(self.no_question_set)])}

        distrs = {k: v for k, v in zip(["reward distrib. from asking question", "reward distrib. from not asking"],
                                       [np.asarray(question_distr), np.asarray(no_question_distr)])}

        return info, distrs

    def reset_epoch_counters(self):

        self.epoch_collects = 0
        self.epoch_redundant_collects = 0

    def update_epoch_gather_stats(self, new_type, old_type, idx):

        if new_type > 0 and old_type == 0:
            self.question_set.add(idx)
            self.no_question_set.remove(idx)
        elif new_type == 0 and old_type > 0:
            self.question_set.remove(idx)
            self.no_question_set.add(idx)
        elif new_type > 0 and old_type < 0:
            self.question_set.add(idx)
        elif new_type == 0 and old_type < 0:
            self.no_question_set.add(idx)

        self.epoch_collects += 1
        if old_type > -1:
            self.epoch_redundant_collects += 1

    def get_unique_captions(self):

        total_unique_caps, total_unique_ask_caps, total_best_unique, total_best_ask_unique = 0, 0, 0, 0

        for imgId, idxs in self.imgId2idx.items():

            temp_cap_set = set()
            temp_ask_cap_set = set()
            temp_best_cap_set = set()
            temp_best_ask_cap_set = set()

            for idx in idxs:

                item = self.collected_data[idx]
                temp_cap_set.update(item['cap_dict'].keys())
                temp_ask_cap_set.update(item['ask_cap_dict'].keys())
                temp_best_cap_set.update(item['best_cap_dict'].keys())
                temp_best_ask_cap_set.update(item['best_ask_cap_dict'].keys())

            total_unique_caps += len(temp_cap_set)
            total_unique_ask_caps += len(temp_ask_cap_set)
            total_best_unique += len(temp_best_cap_set)
            total_best_ask_unique += len(temp_best_ask_cap_set)

        return total_unique_caps, total_unique_ask_caps, total_best_unique, total_best_ask_unique

    def set_gather_stats(self, stats):

        num_scored, num_asked, num_best_scored, num_best_asked = self.get_unique_captions()
        self.collection_stats['num scored'] = num_scored
        self.collection_stats['num asked'] = num_asked
        self.collection_stats['num best scored'] = num_best_scored
        self.collection_stats['num best asked'] = num_best_asked

        for key, value in stats.iteritems():
            self.collection_stats[key] = value

        self.collection_stats['supervision (1, 1, 10)'] = self.collection_stats['num best scored'] + self.collection_stats['num best scored'] + 10 * self.collection_stats['num gt']
        self.collection_stats['supervision (1, 1, 5)'] = self.collection_stats['num best scored'] + self.collection_stats['num best scored'] + 5 * self.collection_stats['num gt']
        self.collection_stats['supervision (1, 0, 10)'] = self.collection_stats['num best scored'] + 10 * self.collection_stats['num gt']
        self.collection_stats['supervision (1, 0, 5)'] = self.collection_stats['num best scored'] + 5 * self.collection_stats['num gt']

    def get_gather_stats(self):
        return self.collection_stats
