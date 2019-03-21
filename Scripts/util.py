import os
import pickle
import torch
from time import time
from torch.utils.data import DataLoader
import argparse
import numpy as np
import random

import Utils.util as util
from Utils.tensorboardlogger import TensorboardLogger
from Utils.visualizer import CaptionVisualizer
from Losses.loss import masked_CE
from datasets.dataset_lifelong import LifelongDataset, DataCollector
from datasets.dataset_caption import CapDataset

from eval.coco_eval import eval_coco, EvalDataset
from Models.attention_captioner import AttentionCaptioner
from Losses.metrics import linear_reward_weighting, get_scorers
from train_caption import validate_helper
from abc import ABCMeta, abstractmethod
from Utils.util import Bunch
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment', type=str)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    return args


def load_model(path, model_class):

    save_state = torch.load(path, map_location=lambda storage, loc: storage)
    model = model_class(save_state['opt']).to(device)
    model.load_state_dict(save_state['state_dict'])

    print '{} model loaded at {}'.format(model_class, path)
    return model


def accum_train_files(round, newfile, oldfile, outfile):
    print("Merging data files {} with {}. Saving to {}".format(newfile, oldfile, outfile))

    d1 = pickle.load(open(newfile, 'rb'))
    d2 = pickle.load(open(oldfile, 'rb'))

    if round == 0:
        for item in d2['data']:
            caption = item['captions'][item['cap_index']]
            item['caption'], item['caption_id'] = caption['caption'], caption['caption_id']
            item['weight'] = d1['gt_caps_reward']

    d1['data'].extend(d2['data'])
    pickle.dump(d1, open(outfile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def replace_word_in_caption(caption, answers, idxs, ask_flag):
    # replace word at index "idxs" in caption with "answers" iff "ask_flag" is set to 1
    caption = caption.clone()

    xcoord = (ask_flag.detach().nonzero()).squeeze()
    if len(xcoord.size()) > 0:
        ycoord = idxs[xcoord]
        vals = answers[xcoord]

        caption[xcoord, ycoord] = vals
    return caption


def get_rollout_replace_stats(replace_reward, rollout_reward, base_reward):
    # get how often rollout caption is better than replace caption and vice versa
    stat_rero = np.sum(replace_reward > rollout_reward)
    stat_rore = np.sum(rollout_reward > replace_reward)
    stat_reall = np.sum(replace_reward > np.maximum(rollout_reward, base_reward))
    stat_roall = np.sum(rollout_reward > np.maximum(replace_reward, base_reward))

    return stat_rero, stat_rore, stat_reall, stat_roall


def choose_better_caption(score1, cap1, mask1, score2, cap2, mask2):
    # given two arrays of captions, and their scores, return [max(cap1[i], cap2[i])] i=1 to length
    caption = []
    cap_mask = []
    for idx, flag in enumerate(score1 >= score2):
        if flag:
            caption.append(cap1[idx].unsqueeze(0))
            cap_mask.append(mask1[idx].unsqueeze(0))
        else:
            caption.append(cap2[idx].unsqueeze(0))
            cap_mask.append(mask2[idx].unsqueeze(0))

    return torch.cat(caption, dim=0), torch.cat(cap_mask, dim=0)


def masked_softmax(logit, cap_mask, valid_pos_mask, temperature=1.0, max_len=17):
    # The <EOS> token represents "don't ask"
    # We have a 1-off problem if caption is actually max+1 length (17) because the last token isn't EOS
    cap_mask, valid_pos_mask = cap_mask.clone(), valid_pos_mask.clone()
    length = torch.clamp(torch.sum(cap_mask != 0, dim=1).long().unsqueeze(1), max=max_len)

    mask = cap_mask * valid_pos_mask
    mask = mask.scatter_(1, length, 1)  # allow asking at the len+1 spot, this is the "dont ask" option

    filter = (mask == 0).clone().float()
    filter[mask==0] = float('-inf')
    masked_prob = torch.softmax((1 / temperature) *logit + filter, dim=1)
    return masked_prob


class LLBaseTrainer(object):
    def __init__(self, args):

        __metaclass__ = ABCMeta
        self.opt = Bunch(json.load(open(args.experiment, 'r')))
        self.opt.resume = args.resume

        self.set_seed(self.opt.seed)
        self.chunk, self.collection_epoch, self.cap_epoch, self.collection_steps, self.cap_steps = 0, 0, 0, 0, 0
        self.start_time = time()

        # LOGGING
        self.result_path = os.path.join(self.opt.exp_dir, self.opt.result_folder, self.opt.run_name)
        util.create_folder(self.result_path)

        self.logger = TensorboardLogger(os.path.join(self.opt.exp_dir, self.opt.result_folder, "tensorboard/" + self.opt.run_name))
        self.std_logger = util.get_std_logger('results', os.path.join(self.result_path, 'stdout.log'))
        self.Cvisualizer = CaptionVisualizer(os.path.join(self.result_path, "{}_cap.html".format(self.opt.run_name)), os.path.join(self.opt.exp_dir, "Utils/css/caption.css"))

        self.init_files_dirs()

        # Auto-eval scorers
        self.scorers = get_scorers(self.opt.cached_tokens)

        self.get_data_loaders()

        self.captioner = load_model(self.opt.cap_path, AttentionCaptioner)

        self.c_optimizer = torch.optim.Adam(
            [
                {'params': filter(lambda p: p.requires_grad, self.captioner.parameters()), 'lr': self.opt.c_lr,
                 'mod_name': 'cap'}
            ])

        if self.opt.resume is not None:
            self.resume(self.opt.resume)

    @abstractmethod
    def loop_chunk(self, repochs):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def load_previous_chunk_best_models(self):
        pass

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def init_files_dirs(self):
        base = self.result_path
        self.lifelong_data_files = self.opt.train_files.split(',')
        self.chunks = len(self.lifelong_data_files)

        self.cap_checkpoint_path, self.collection_path, self.dm_checkpoint_path \
            = [base+x for x in ["/ccheckpoints", "/collected_data", "/mcheckpoints"]]

        for d in [self.cap_checkpoint_path, self.collection_path, self.dm_checkpoint_path]:
            util.create_folder(d)

        self.collected_data_paths = [os.path.join(self.collection_path, 'CollectedDataRound{}.p'.format(x+1))
                                     for x in range(self.chunks)]
        self.caption_data_files = [self.opt.warmup_file]
        self.caption_data_files.extend([os.path.join(self.collection_path, 'AccumDataRound{}.p'.format(x+1))
                                        for x in range(self.chunks)])
        self.chunks = len(self.lifelong_data_files)
        self.caption_model_files = [os.path.join(self.cap_checkpoint_path, 'chunk{}_best.pth'.format(x+1))
                                    for x in range(self.chunks)]
        self.decision_maker_model_files = [os.path.join(self.dm_checkpoint_path, 'chunk{}_best.pth'.format(x+1))
                                    for x in range(self.chunks)]

    def get_data_loaders(self):

        val_data = CapDataset(self.opt, self.opt.val_file)
        self.val_loader = DataLoader(val_data, batch_size=self.opt.batch_size, num_workers=1, shuffle=False)

        eval_dataset = EvalDataset(self.opt, self.opt.eval_file, split="val")
        self.eval_loader = DataLoader(eval_dataset, batch_size=self.opt.batch_size, num_workers=1, shuffle=False)

        self.c_i2w, self.p_i2w, self.q_i2w, self.a_i2w = val_data.c_i2w, val_data.p_i2w, val_data.q_i2w, val_data.a_i2w

        self.opt.c_vocab_size, self.opt.pos_vocab_size, self.opt.q_vocab_size, self.opt.a_vocab_size = \
            val_data.c_vocab_size, val_data.pos_vocab_size, val_data.q_vocab_size, val_data.a_vocab_size

        self.opt.special_symbols, self.special_symbols = val_data.special_symbols, val_data.special_symbols

        self.std_logger.info("Size of caption vocab: {} \n Size of pos vocab: {} \nMax length of captions: {}"
                             .format(self.opt.c_vocab_size, self.opt.pos_vocab_size, val_data.max_c_len))

    def update_lr(self, epoch):
        if self.opt.update_lr:
            if epoch > self.opt.learning_rate_decay_start:
                new_lr = util.update_lr(epoch, self.opt.c_lr, self.opt.learning_rate_decay_start, self.opt.learning_rate_decay_every, self.opt.learning_rate_decay_rate)
                util.set_lr(self.c_optimizer, new_lr)
                self.std_logger.info("Setting lr to {}".format(new_lr))

    def update_ss(self, epoch):
        if self.opt.c_scheduled_sampling and epoch > self.opt.c_scheduled_sampling_start:
            self.captioner.ss_prob = util.update_scheduled_sampling(epoch, self.opt.c_scheduled_sampling_start,
                                                           self.opt.c_scheduled_sampling_increase_every,
                                                           self.opt.c_scheduled_sampling_increase_prob,
                                                           self.opt.c_scheduled_sampling_max_prob)
            self.std_logger.info("Setting word scheduled sampling to {}".format(self.captioner.ss_prob))

        if self.opt.p_scheduled_sampling and epoch > self.opt.p_scheduled_sampling_start:
            self.captioner.decoder.pos_ss_prob = util.update_scheduled_sampling(epoch, self.opt.p_scheduled_sampling_start,
                                                                       self.opt.p_scheduled_sampling_increase_every,
                                                                       self.opt.p_scheduled_sampling_increase_prob,
                                                                       self.opt.p_scheduled_sampling_max_prob,
                                                                       self.opt.p_scheduled_sampling_initial_value)
            self.std_logger.info("Setting pos scheduled sampling to {}".format(self.captioner.decoder.pos_ss_prob))

    def pad_caption(self, caption, cap_len):
        # replace 0's beyond caption_length in captions with pad symbol int(c_vocab_size)
        padding = self.range_matrix[:caption.size(0)] >= cap_len.unsqueeze(1).repeat(1, self.opt.c_max_sentence_len+1)
        caption = caption + padding.long() * self.opt.c_vocab_size
        return caption

    def pad_question(self, question, que_mask):
        # replace 0's beyond question_length in question with pad symbol int(q_vocab_size)
        return question * que_mask.long() \
                   + (self.opt.q_vocab_size * self.ones_matrix[:question.size(0)]) * (que_mask == 0).long()

    def reset_captioner(self, round=None):
        if round is None:
            print("Resetting captioner to random weights.")
            self.captioner.apply(util.weights_init)
        else:
            print("Loading captioner from {}.".format(self.caption_model_files[round]))
            save_state = torch.load(self.caption_model_files[round], map_location=lambda storage, loc: storage)
            self.captioner.load_state_dict(save_state['state_dict'])

    def get_next_round_data(self, train_file):
        print("Loaded training data from {}.".format(train_file))
        self.data_collector = DataCollector(self.opt, train_file)
        train_data = CapDataset(self.opt, train_file)
        self.train_loader = DataLoader(train_data, batch_size=self.opt.batch_size, num_workers=1, shuffle=True)

    def get_caption_train_data(self, train_file):
        print("Loaded training data from {}.".format(train_file))
        train_data = LifelongDataset(self.opt, train_file)
        self.train_loader = DataLoader(train_data, batch_size=self.opt.batch_size, num_workers=1, shuffle=True)

    def loop_lifelong(self):
        print("Lifelong training")

        for chunk in range(self.chunk, self.chunks):
            self.chunk = chunk

            self.best_d_score = float('-inf')
            self.best_c_score = float('-inf')

            # Load the unlabelled images for this chunk
            self.get_next_round_data(self.lifelong_data_files[self.chunk])

            # Put models into train or eval based on config
            if self.opt.cap_eval:
                self.captioner.eval()
            else:
                self.captioner.train()

            # Loop over one chunk of data and collect captions
            self.loop_chunk(self.opt.epochs_per_chunk)

            # Save collected captions
            self.data_collector.save_collected_data(self.collected_data_paths[chunk], self.std_logger)

            # Logging
            stats = self.data_collector.get_gather_stats()
            util.round_logging(self.logger, stats, self.chunk+1)

            # Accumulate caption dataset
            accum_train_files(round=self.chunk, newfile=self.collected_data_paths[chunk], oldfile=self.caption_data_files[chunk], outfile=self.caption_data_files[chunk+1])

            # Reset captioner weights
            if self.opt.reinit_cap_weights:
                self.reset_captioner(round=None)
            else:
                self.reset_captioner(round=self.chunk - 1)

            # Train captioner on accumulated data
            self.get_caption_train_data(self.caption_data_files[self.chunk+1])
            self.train_captioner()

            # Load best captioner and decision maker models for next chunk
            self.load_previous_chunk_best_models()

            self.save_checkpoint(chunk)

    def train_captioner(self):

        self.captioner.train()

        for epoch in range(self.opt.cap_epochs):
            self.cap_epoch = epoch

            self.update_lr(epoch)
            self.update_ss(epoch)

            print_loss, tic = 0, time()

            print("Training captioner")

            for i, sample in enumerate(self.train_loader):

                image, source, target, caption_len, pos, weight = [x.to(device) for x in sample]

                # Forward pass
                self.c_optimizer.zero_grad()

                r = self.captioner(image, source, pos)
                logits, pos_logits = r.logits, r.pos_logits

                if self.opt.weight_captions:
                    word_loss = masked_CE(logits, target, caption_len, weight.float())
                    pos_loss = masked_CE(pos_logits, pos, caption_len-1, weight.float())
                else:
                    word_loss = masked_CE(logits, target, caption_len)
                    pos_loss = masked_CE(pos_logits, pos, caption_len-1)

                total_loss = word_loss + self.opt.pos_alpha * pos_loss

                # Backwards pass
                total_loss.backward()

                if self.opt.grad_clip:
                    util.gradient_noise_and_clip(self.captioner.parameters(), self.opt.max_clip)

                self.c_optimizer.step()

                # Logging
                print_loss += total_loss.item()

                if self.cap_steps % self.opt.print_every == 0:
                    info = {
                        'cap/loss': print_loss/self.opt.print_every,
                        'cap/time': (time() - tic)/self.opt.print_every  # total time so far for this epoch
                    }
                    util.step_logging(self.logger, info, self.cap_steps)
                    util.log_avg_grads(self.logger, self.captioner, self.cap_steps, name="cap/")
                    steps_per_epoch = len(self.train_loader)
                    self.std_logger.info(
                        "Chunk {} Epoch {}, {}/{}| Loss: {} | Time per batch: {} |"
                        " Epoch remaining time (HH:MM:SS) {} | Elapsed time {}"
                        .format(self.chunk+1, epoch+1, i, steps_per_epoch, info['cap/loss'], info['cap/time'],
                                util.time_remaining(steps_per_epoch - i, info['cap/time']),
                                util.time_elapsed(self.start_time, time())))

                    print_loss, tic = 0, time()

                self.cap_steps += 1

            model_score = self.evaluate_captioner()
            self.save_captioner(epoch, model_score)

    def save_captioner(self, epoch, model_score):

        save_state = {
            'epoch': epoch+1,
            'cap_steps': self.cap_steps,
            'cap_epoch': self.cap_epoch,
            'state_dict': self.captioner.state_dict(),
            'optimizer': self.c_optimizer.state_dict(),
            'opt': self.opt,
            'best_score': self.best_c_score
        }

        if model_score > self.best_c_score:
            self.std_logger.info("New best captioner score {} > {} previous score.".format(model_score, self.best_c_score))
            save_name = self.caption_model_files[self.chunk]
            torch.save(save_state, save_name)
            self.best_c_score = model_score

    def save_checkpoint(self, chunk):

        save_state = {
            'chunk': chunk+1,
            'cap_steps': self.cap_steps,
            'collection_steps': self.collection_steps,
            'cap_epoch': self.cap_epoch,
            'collection_epoch': self.collection_epoch,
            'opt': self.opt
        }

        save_name = os.path.join(self.result_path, '{}_checkpoint.pth'.format(self.chunk+1))
        torch.save(save_state, save_name)

    def resume(self, path):
        save_state = torch.load(path, map_location=lambda storage, loc: storage)

        self.chunk = save_state['chunk']
        self.cap_steps = save_state['cap_steps']
        self.collection_steps = save_state['collection_steps']
        self.cap_epoch = save_state['cap_epoch']
        self.collection_epoch = save_state['collection_epoch']

        self.load_previous_chunk_best_models()

        print 'Model reloaded to resume from chunk {} at {}'.format(self.chunk, path)

    def evaluate_captioner(self):

        print("Validating captioner")

        # compute loss, word-for-word accuracy, and coco-caption metrics
        val_loss, val_acc, val_pos_acc, val_pos_loss = self.validate_captioner()
        scores = eval_coco(self.captioner, self.eval_loader, self.opt.run_name, self.result_path, self.opt)
        weighted_score = linear_reward_weighting(scores['Bleu_1'], scores['Bleu_2'], scores['Bleu_3'], scores['Bleu_4'],
                                             scores['ROUGE_L'], scores['METEOR'], scores['CIDEr']) * 100.0
        model_score = weighted_score

        info = {
            'val loss': val_loss,
            'val accuracy': val_acc,
            'val pos accuracy': val_pos_acc,
            'val pos loss': val_pos_loss,
            'eval cider': scores['CIDEr']*100.0,
            'eval bleu 4': scores['Bleu_4']*100.0,
            'eval bleu 3': scores['Bleu_3']*100.0,
            'eval bleu 2': scores['Bleu_2']*100.0,
            'eval bleu 1': scores['Bleu_1']*100.0,
            'eval rouge L': scores['ROUGE_L']*100.0,
            'eval meteor': scores['METEOR']*100.0,
            'eval weighted score': weighted_score
        }

        self.std_logger.info(str(info))
        util.epoch_logging(self.logger, info, self.chunk*self.opt.cap_epochs+self.cap_epoch)

        return model_score

    def validate_captioner(self):

        self.captioner.eval()
        word_loss, word_cor, pos_cor, pos_loss = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for step, sample in enumerate(self.val_loader):
                image, source, target, caption_len, refs, ref_lens, pos, img_path = sample[:-2]
                sample = [x.to(device) for x in [image, source, target, caption_len, pos]]
                r = validate_helper(sample, self.captioner, self.val_loader.dataset.max_c_len)

                word_loss, word_cor, pos_cor, pos_loss = [x+y for x,y in zip([word_loss, word_cor, pos_cor, pos_loss], r)]

                # write image and predicted captions to html file to visualize training progress
                if step % self.opt.val_print_every == 0:
                    beam_size = 3
                    c_i2w = self.val_loader.dataset.c_i2w
                    p_i2w = self.val_loader.dataset.p_i2w
                    refs, ref_lens = [util.to_np(x) for x in [refs, ref_lens]]

                    # show top 3 beam search captions
                    result = self.captioner.sample_beam(sample[0], beam_size=beam_size, max_seq_len=17)
                    captions, lps, lens = [util.to_np(x) for x in [result.captions, torch.sum(result.logprobs, dim=2),
                                                                   torch.sum(result.logprobs.abs() > 0.00001, dim=2)]]

                    # show greedy caption
                    result = self.captioner.sample(sample[0], greedy=True, max_seq_len=17)
                    pprob, pidx = torch.max(result.pos_prob, dim=2)
                    gcap, glp, glens, pidx, pprob = [util.to_np(x) for x in [result.caption, torch.sum(result.log_prob, dim=1),
                                                        torch.sum(result.mask, dim=1), pidx, pprob]]

                    for i in range(image.size(0)):
                        cap_arr = util.idx2str(c_i2w, (gcap[i])[:glens[i]])
                        pos_arr = util.idx2pos(p_i2w, (pidx[i])[:glens[i]])
                        pos_pred = ["{} ({} {:.2f})".format(cap_arr[j], pos_arr[j], pprob[i, j]) for j in range(glens[i])]

                        entry = {
                            'img': img_path[i],
                            'epoch': self.cap_epoch,
                            'greedy_sample': ' '.join(cap_arr) + " logprob: {}".format(glp[i]),
                            'pos_pred': ' '.join(pos_pred),
                            'beamsearch': [
                                ' '.join(util.idx2str(c_i2w, (captions[i, j])[:lens[i, j]])) + " logprob: {}".format(lps[i, j]) for j in range(beam_size)
                            ],
                            'refs': [
                                ' '.join(util.idx2str(c_i2w, (refs[i, j])[:ref_lens[i, j]])) for j in range(3)
                            ]
                        }

                        self.Cvisualizer.add_entry(entry)

        self.Cvisualizer.update_html()
        # Reset
        self.captioner.train()

        return [x / len(self.val_loader.dataset) for x in [word_loss, word_cor, pos_cor, pos_loss]]
