import os
import json
import argparse
from time import time
import numpy as np
import random

import torch
from torch.utils.data import DataLoader
from Models.attention_questioner import QuestionGenerator
from datasets.dataset_quegen import WordMatchDataset, PosMatchDataset
from Utils.tensorboardlogger import TensorboardLogger
import Utils.util as util
from Losses.loss import masked_CE, seq_max_and_mask
from Models.attention_vqa import AttentionVQA
from Models.attention_captioner import AttentionCaptioner
from Utils.util import Bunch
from Utils.visualizer import QGenVisualizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
random.seed(1)

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


class trainer:
    def __init__(self, args):

        self.opt = Bunch(json.load(open(args.experiment, 'r')))
        self.opt.resume = args.resume

        self.global_step = 0
        self.epoch = 0
        self.best_score = 0.0
        self.start_time = time()

        # LOGGING
        self.result_path = os.path.join(self.opt.exp_dir, self.opt.result_folder, self.opt.run_name)
        util.create_folder(self.result_path)
        util.create_folder(os.path.join(self.result_path, 'checkpoints'))
        self.logger = TensorboardLogger(os.path.join(self.opt.exp_dir, self.opt.result_folder,
                                                     "tensorboard/" + self.opt.run_name))
        self.std_logger = util.get_std_logger('results', os.path.join(self.result_path, 'stdout.log'))
        self.visualizer = QGenVisualizer(os.path.join(self.result_path, "{}.html".format(self.opt.run_name)),
                                         os.path.join(self.opt.exp_dir, "Utils/css/question.css"))

        self.get_data_loaders()

        # MODEL
        self.model = QuestionGenerator(self.opt).to(device)
        self.model.apply(util.weights_init)

        self.vqa = load_model(self.opt.vqa_path, AttentionVQA)
        self.captioner = load_model(self.opt.cap_path, AttentionCaptioner)

        self.vqa.eval()
        if self.opt.cap_eval:
            self.captioner.eval()

        # OPTIMIZER
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opt.lr)

        if self.opt.resume is not None:
            self.resume(self.opt.resume)

        self.arang_vector = torch.arange(self.opt.batch_size, dtype=torch.long).to(device)

    def save_checkpoint(self, epoch, model_score):

        save_state = {
            'epoch': epoch+1,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'opt': self.opt,
            'best_score': self.best_score
        }

        if self.opt.checkpoint_every_epoch:
            save_name = os.path.join(self.result_path, 'checkpoints', '{}_epoch{}.pth'.format(self.opt.run_name, epoch+1))
            torch.save(save_state, save_name)
        else:
            save_name = os.path.join(self.result_path, 'checkpoints', '{}_last.pth'.format(self.opt.run_name))
            torch.save(save_state, save_name)

        if model_score > self.best_score:
            self.std_logger.info("New best model score {} > {} previous score.".format(model_score, self.best_score))
            save_name = os.path.join(self.result_path, 'checkpoints', '{}_best.pth'.format(self.opt.run_name))
            torch.save(save_state, save_name)
            self.best_score = model_score

    def resume(self, path):

        save_state = torch.load(path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(save_state['state_dict'])
        self.global_step = save_state['global_step']
        self.epoch = save_state['epoch']
        self.optimizer.load_state_dict(save_state['optimizer'])
        self.best_score = save_state['best_score']

        print 'Model reloaded to resume from Epoch %d, Global Step %d from model at %s'%(self.epoch, self.global_step, path)

    def get_data_loaders(self):

        word_match_data = WordMatchDataset(self.opt, self.opt.train_word_file)
        self.word_match_loader = DataLoader(word_match_data, batch_size=self.opt.batch_size/2, num_workers=1, shuffle=True)

        pos_match_data = PosMatchDataset(self.opt, self.opt.train_pos_file, fake_len=len(word_match_data))
        self.pos_match_loader = DataLoader(pos_match_data, batch_size=self.opt.batch_size/2, num_workers=1, shuffle=True)

        val_data = WordMatchDataset(self.opt, self.opt.val_file)
        self.val_loader = DataLoader(val_data, batch_size=self.opt.batch_size, num_workers=1, shuffle=False)

        self.opt.c_vocab_size, self.opt.q_vocab_size, self.opt.p_vocab_size, self.opt.special_symbols \
            = word_match_data.c_vocab_size, word_match_data.q_vocab_size, word_match_data.p_vocab_size, word_match_data.special_symbols

        self.c_i2w, self.p_i2w, self.q_i2w = word_match_data.c_i2w, word_match_data.p_i2w, word_match_data.q_i2w
        self.opt.c_i2w, self.opt.p_i2w, self.opt.q_i2w = self.c_i2w, self.p_i2w, self.q_i2w

        self.std_logger.info("Size of question vocab: {} \nSize of pos vocab: {} \nMax length of questions: {}"
              .format(self.opt.q_vocab_size, self.opt.p_vocab_size, word_match_data.max_q_len))

    def update_lr(self, epoch):
        if self.opt.update_lr:
            if epoch > self.opt.learning_rate_decay_start:
                new_lr = util.update_lr(epoch, self.opt.lr, self.opt.learning_rate_decay_start, self.opt.learning_rate_decay_every, self.opt.learning_rate_decay_rate)
                util.set_lr(self.optimizer, new_lr)
                self.std_logger.info("Setting lr to {}".format(new_lr))

    def update_ss(self, epoch):
        if self.opt.scheduled_sampling and epoch > self.opt.scheduled_sampling_start:
            self.model.ss_prob = util.update_scheduled_sampling(epoch, self.opt.scheduled_sampling_start,
                                                           self.opt.scheduled_sampling_increase_every,
                                                           self.opt.scheduled_sampling_increase_prob,
                                                           self.opt.scheduled_sampling_max_prob)
            self.std_logger.info("Setting word scheduled sampling to {}".format(self.model.ss_prob))

    def compute_cap_features(self, word_batch, pos_batch):
        match_bs = word_batch[0].size(0)
        unmatch_bs = pos_batch[0].size(0)

        image = torch.cat([word_batch[0], pos_batch[0]], dim=0)
        caption = torch.cat([word_batch[4], pos_batch[4]], dim=0)
        result = self.captioner(image, caption, gt_pos=None, ss=False)

        question_len, source, target, q_idx, q_idx_vec = [torch.cat([word_batch[x], pos_batch[x]], dim=0) for x in [1, 2, 3, 5, 6]]
        caption = caption[:, 1:]  # qgen doesn't need to encode the bos token
        q_idx_vec = q_idx_vec.float()

        pos_logits = result.pos_logits[self.arang_vector[:match_bs + unmatch_bs], q_idx].detach()
        pos_probs = torch.softmax(pos_logits, dim=1)

        attm = result.att[self.arang_vector[:match_bs], q_idx[:match_bs]].detach()
        attunm = result.att[self.arang_vector[match_bs:match_bs + unmatch_bs]].detach()
        attunm = torch.softmax(attunm.sum(dim=1), dim=1)  # use the average attention vector, renormalize
        att = torch.cat([attm, attunm], dim=0)

        hidm = result.hidden[self.arang_vector[:match_bs], q_idx[:match_bs]].detach()
        hidunm = torch.normal(torch.zeros((unmatch_bs, hidm.size(1))), 0.5 * torch.ones(unmatch_bs, hidm.size(1))).to(device)
        hidden = torch.cat([hidm, hidunm], dim=0)

        return [image, question_len, source, target, caption, q_idx_vec, pos_probs, att, hidden]

    def loop(self):
        for epoch in range(self.epoch, self.opt.max_epochs):
            self.epoch = epoch

            self.update_lr(epoch)
            self.update_ss(epoch)
            self.train(epoch)

    def train(self, epoch):

        print("Training")

        print_loss, tic = 0, time()
        self.model.train()

        # manually iterate over dataset
        word_iter = self.word_match_loader.__iter__()
        pos_iter = self.pos_match_loader.__iter__()
        while True:
            try:
                word_batch = word_iter.next()
                pos_batch = pos_iter.next()
            except StopIteration:
                break

            word_batch = [x.to(device) for x in word_batch[:-1]]
            pos_batch = [x.to(device) for x in pos_batch[:-1]]

            image, question_len, source, target, caption, q_idx_vec, pos, att, context = self.compute_cap_features(word_batch, pos_batch)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(image, caption, pos, context, att, source, q_idx_vec)
            loss = masked_CE(logits, target, question_len)

            # Backward pass
            loss.backward()

            if self.opt.grad_clip:
                util.gradient_noise_and_clip(self.model.parameters(), self.opt.max_clip)

            self.optimizer.step()

            # Logging
            print_loss += loss.item()

            if self.global_step % self.opt.print_every == 0:
                info = {
                    'loss': print_loss/self.opt.print_every,
                    'time': (time() - tic)/self.opt.print_every  # time per step
                }

                util.step_logging(self.logger, info, self.global_step)
                util.log_avg_grads(self.logger, self.model, self.global_step)

                steps_per_epoch = len(self.word_match_loader)
                step = self.global_step - epoch*steps_per_epoch
                remaining_steps = steps_per_epoch*(self.opt.max_epochs-epoch)-step
                self.std_logger.info("{}, {}/{}| Loss: {} | Time per batch: {} | Epoch remaining time (HH:MM:SS) {} | "
                                     "Elapsed time {} | Total remaining time {}"
                                     .format(epoch+1, step, steps_per_epoch, info['loss'], info['time'],
                                    util.time_remaining(steps_per_epoch-step, info['time']),
                                    util.time_elapsed(self.start_time, time()),
                                    util.time_remaining(remaining_steps, info['time'])))
                print_loss, tic = 0, time()

            self.global_step = self.global_step + 1

        model_score = self.evaluate(epoch+1)
        self.save_checkpoint(epoch, model_score)

    def evaluate(self, epoch):

        val_loss, val_acc, val_greedy_correct = self.validate()
        model_score = val_greedy_correct

        info = {
            'val loss': val_loss,
            'val accuracy': val_acc,
            'val correct answers': val_greedy_correct
        }

        self.std_logger.info(str(info))

        util.epoch_logging(self.logger, info, epoch)
        # util.parameter_logging(self.logger, self.model, epoch)

        return model_score

    def compute_cap_features_val(self, batch):
        bs = batch[0].size(0)

        image, caption = batch[0], batch[4]
        result = self.captioner(image, caption, gt_pos=None, ss=False)

        question_len, source, target, q_idx, q_idx_vec, refs, answer = batch[1], batch[2], batch[3], batch[5], batch[6], batch[7], batch[8]
        q_idx_vec = q_idx_vec.float()
        caption = caption[:, 1:]

        pos_logits = result.pos_logits[self.arang_vector[:bs], q_idx].detach()
        pos_probs = torch.softmax(pos_logits, dim=1)
        att = result.att[self.arang_vector[:bs], q_idx].detach()
        hidden = result.hidden[self.arang_vector[:bs], q_idx].detach()

        return [image, question_len, source, target, caption, q_idx_vec, pos_probs, att, hidden, refs, answer]

    def query_vqa(self, image, question, captions, answer, mask):

        pad_symbol = self.opt.q_vocab_size

        inverse_mask = mask == 0
        question = question * mask.long() + (pad_symbol * torch.ones(mask.size(), dtype=torch.long)).to(
            device) * inverse_mask.long()

        result = self.vqa(image, question, captions)

        _, output_max_index = torch.max(result.probs, 1)

        num_correct = (answer == output_max_index).float().sum().item()

        return num_correct

    def validate(self):

        loss, correct, correct_answers = 0.0, 0.0, 0
        self.model.eval()

        for step, batch in enumerate(self.val_loader):
            img_path = batch[-1]
            batch = [x.to(device) for x in batch[:-1]]

            image, question_len, source, target, caption, q_idx_vec, pos, att, context, refs, answer = self.compute_cap_features_val(batch)

            logits = self.model(image, caption, pos, context, att, source, q_idx_vec)

            batch_loss = masked_CE(logits, target, question_len)
            loss += batch_loss.item()
            predictions = seq_max_and_mask(logits, question_len, self.val_loader.dataset.max_q_len+1)

            correct += torch.sum((target == predictions).float() / question_len.float().unsqueeze(1)).item()

            # evaluate using VQA expert
            result = self.model.sample(image, caption, pos, context, att, q_idx_vec, greedy=True, max_seq_len=self.val_loader.dataset.max_q_len+1)
            sample, log_probs, mask = result.question, result.log_prob, result.mask

            correct_answers += self.query_vqa(image, sample, refs, answer, mask)

            # write image and Q&A to html file to visualize training progress
            if step % self.opt.visualize_every == 0:
                beam_size = 3
                c_i2w = self.val_loader.dataset.c_i2w
                a_i2w = self.val_loader.dataset.a_i2w
                q_i2w = self.val_loader.dataset.q_i2w

                sample_len = mask.sum(dim=1)

                _, _, beam_predictions, beam_lps = self.model.sample_beam(image, caption, pos, context, att, q_idx_vec, beam_size=beam_size, max_seq_len=15)
                beam_predictions, lps, lens = [util.to_np(x)
                                                         for x in [beam_predictions, torch.sum(beam_lps, dim=2), torch.sum(beam_lps != 0, dim=2)]]

                target, question_len, sample, sample_len = [util.to_np(x) for x in [target, question_len, sample, sample_len]]

                for i in range(image.size(0)):

                    entry = {
                        'img': img_path[i],
                        'epoch': self.epoch,
                        'answer': a_i2w[answer[i].item()],
                        'gt_question': ' '.join(util.idx2str(q_i2w, (target[i])[:question_len[i]-1])),
                        'greedy_question': ' '.join(util.idx2str(q_i2w, (sample[i])[:sample_len[i]])),
                        'beamsearch': [
                            ' '.join(util.idx2str(q_i2w, (beam_predictions[i, j])[:lens[i, j]])) + " logprob: {}".format(lps[i, j]) for j in range(beam_size)
                        ]
                    }
                    self.visualizer.add_entry(entry)

        self.visualizer.update_html()

        # Reset
        self.model.train()

        l = len(self.val_loader.dataset)
        return [loss / l, correct / l, 100.0*correct_answers / l]


if __name__ == "__main__":

    args = get_args()
    t = trainer(args)
    t.loop()
