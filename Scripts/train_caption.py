import os
import json
from time import time
import argparse
import torch
from torch.utils.data import DataLoader
from eval.coco_eval import eval_coco, EvalDataset
from Models.attention_captioner import AttentionCaptioner
from datasets.dataset_caption import CapDataset
from Utils.tensorboardlogger import TensorboardLogger
import Utils.util as util
from Losses.loss import masked_CE, seq_max_and_mask
from Losses.metrics import get_scorers, linear_reward_weighting
from Utils.visualizer import CaptionVisualizer
from Utils.util import Bunch
import numpy as np
import random

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
        self.logger = TensorboardLogger(os.path.join(self.opt.exp_dir, self.opt.result_folder, "tensorboard/" + self.opt.run_name))
        self.std_logger = util.get_std_logger('results', os.path.join(self.result_path, 'stdout.log'))
        self.visualizer = CaptionVisualizer(os.path.join(self.result_path, "{}.html".format(self.opt.run_name)), os.path.join(self.opt.exp_dir, "Utils/css/caption.css"))

        self.get_data_loaders()

        # MODEL
        self.model = AttentionCaptioner(self.opt).to(device)
        self.model.apply(util.weights_init)

        # OPTIMIZER
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opt.lr)

        # Auto-eval scorers
        self.scorers = get_scorers(self.opt.cached_tokens)

        if self.opt.resume is not None:
            self.resume(self.opt.resume)

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

        train_data = CapDataset(self.opt, self.opt.train_file)
        val_data = CapDataset(self.opt, self.opt.val_file)

        self.train_loader = DataLoader(train_data, batch_size=self.opt.batch_size, num_workers=1, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=self.opt.batch_size, num_workers=1, shuffle=False)

        eval_dataset = EvalDataset(self.opt, self.opt.eval_file, split="val")
        self.eval_loader = DataLoader(eval_dataset, batch_size=self.opt.batch_size, num_workers=1, shuffle=False)

        self.opt.c_vocab_size, self.opt.pos_vocab_size, self.opt.special_symbols = \
            train_data.c_vocab_size, train_data.pos_vocab_size, train_data.special_symbols

        self.std_logger.info("Size of question vocab: {} \n Size of pos vocab: {} \nMax length of questions: {}"
                             .format(self.opt.c_vocab_size, self.opt.pos_vocab_size, train_data.max_c_len))

    def update_lr(self, epoch):
        if self.opt.update_lr:
            if epoch > self.opt.learning_rate_decay_start:
                new_lr = util.update_lr(epoch, self.opt.lr, self.opt.learning_rate_decay_start, self.opt.learning_rate_decay_every, self.opt.learning_rate_decay_rate)
                util.set_lr(self.optimizer, new_lr)
                self.std_logger.info("Setting lr to {}".format(new_lr))

    def update_ss(self, epoch):
        if self.opt.c_scheduled_sampling and epoch > self.opt.c_scheduled_sampling_start:
            self.model.ss_prob = util.update_scheduled_sampling(epoch, self.opt.c_scheduled_sampling_start,
                                                           self.opt.c_scheduled_sampling_increase_every,
                                                           self.opt.c_scheduled_sampling_increase_prob,
                                                           self.opt.c_scheduled_sampling_max_prob)
            self.std_logger.info("Setting word scheduled sampling to {}".format(self.model.ss_prob))

        if self.opt.p_scheduled_sampling and epoch > self.opt.p_scheduled_sampling_start:
            self.model.decoder.pos_ss_prob = util.update_scheduled_sampling(epoch, self.opt.p_scheduled_sampling_start,
                                                                       self.opt.p_scheduled_sampling_increase_every,
                                                                       self.opt.p_scheduled_sampling_increase_prob,
                                                                       self.opt.p_scheduled_sampling_max_prob,
                                                                       self.opt.p_scheduled_sampling_initial_value)
            self.std_logger.info("Setting pos scheduled sampling to {}".format(self.model.decoder.pos_ss_prob))

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

        for i, sample in enumerate(self.train_loader):
            image, source, target, caption_len, refs, ref_lens, pos = sample[:-3]
            image, source, target, caption_len, pos = [x.to(device) for x in [image, source, target, caption_len, pos]]

            # Forward pass
            self.optimizer.zero_grad()
            result = self.model(image, source, pos)
            logits, pos_logits = result.logits, result.pos_logits

            # Get losses
            word_loss = masked_CE(logits, target, caption_len)
            pos_loss = masked_CE(pos_logits, pos, caption_len - 1)

            total_loss = word_loss + self.opt.pos_alpha * pos_loss

            # Backward pass
            total_loss.backward()

            if self.opt.grad_clip:
                util.gradient_noise_and_clip(self.model.parameters(), self.opt.max_clip)

            self.optimizer.step()

            loss = total_loss.item()

            # Logging
            print_loss += loss

            if self.global_step % self.opt.print_every == 0:
                info = {
                    'loss': print_loss/self.opt.print_every,
                    'time': (time() - tic)/self.opt.print_every  # time per step
                }
                util.step_logging(self.logger, info, self.global_step)
                util.log_avg_grads(self.logger, self.model, self.global_step)

                steps_per_epoch = len(self.train_loader)
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

        # compute loss, word-for-word accuracy, and coco-caption metrics
        val_loss, val_acc, val_pos_acc, val_pos_loss = self.validate()
        scores = eval_coco(self.model, self.eval_loader, self.opt.run_name, self.result_path, self.opt)
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
        util.epoch_logging(self.logger, info, epoch)

        return model_score

    def validate(self):

        print("Validating")

        self.model.eval()
        word_loss, word_cor, pos_cor, pos_loss = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for step, sample in enumerate(self.val_loader):

                image, source, target, caption_len, refs, ref_lens, pos, img_path = sample[:-2]
                sample = [x.to(device) for x in [image, source, target, caption_len, pos]]

                r = validate_helper(sample, self.model, self.val_loader.dataset.max_c_len)

                word_loss, word_cor, pos_cor, pos_loss = [x+y for x,y in zip([word_loss, word_cor, pos_cor, pos_loss], r)]

                # write image and predicted captions to html file to visualize training progress
                if step % self.opt.visualize_every == 0:
                    beam_size = 3
                    c_i2w = self.val_loader.dataset.c_i2w
                    p_i2w = self.val_loader.dataset.p_i2w
                    refs, ref_lens = [util.to_np(x) for x in [refs, ref_lens]]

                    # show top 3 beam search captions
                    result = self.model.sample_beam(sample[0], beam_size=beam_size, max_seq_len=17)
                    captions, lps, lens = [util.to_np(x) for x in [result.captions, torch.sum(result.logprobs, dim=2),
                                                                   torch.sum(result.logprobs.abs() > 0.00001, dim=2)]]

                    # show greedy caption
                    result = self.model.sample(sample[0], greedy=True, max_seq_len=17)
                    pprob, pidx = torch.max(result.pos_prob, dim=2)
                    gcap, glp, glens, pidx, pprob = [util.to_np(x) for x in [result.caption, torch.sum(result.log_prob, dim=1),
                                                        torch.sum(result.mask, dim=1), pidx, pprob]]

                    for i in range(image.size(0)):
                        cap_arr = util.idx2str(c_i2w, (gcap[i])[:glens[i]])
                        pos_arr = util.idx2pos(p_i2w, (pidx[i])[:glens[i]])
                        pos_pred = ["{} ({} {:.2f})".format(cap_arr[j], pos_arr[j], pprob[i, j]) for j in range(glens[i])]

                        entry = {
                            'img': img_path[i],
                            'epoch': self.epoch,
                            'greedy_sample': ' '.join(cap_arr) + " logprob: {}".format(glp[i]),
                            'pos_pred': ' '.join(pos_pred),
                            'beamsearch': [
                                ' '.join(util.idx2str(c_i2w, (captions[i, j])[:lens[i, j]])) + " logprob: {}".format(lps[i, j]) for j in range(beam_size)
                            ],
                            'refs': [
                                ' '.join(util.idx2str(c_i2w, (refs[i, j])[:ref_lens[i, j]])) for j in range(3)
                            ]
                        }

                        self.visualizer.add_entry(entry)

        self.visualizer.update_html()
        # Reset
        self.model.train()

        return [x / len(self.val_loader.dataset) for x in [word_loss, word_cor, pos_cor, pos_loss]]


def validate_helper(sample, model, max_c_len):

    image, source, target, caption_len, pos = sample

    # Compute loss

    result = model(image, source, pos)
    logits, pos_logits = result.logits, result.pos_logits

    word_loss = masked_CE(logits, target, caption_len).item()
    pos_loss = masked_CE(pos_logits, pos, caption_len-1).item()

    # Compute symbol accuracy

    word_preds = seq_max_and_mask(logits, caption_len, max_c_len + 1)
    pos_preds = seq_max_and_mask(pos_logits, caption_len-1, max_c_len + 1)

    cor = torch.sum((target == word_preds).float() / caption_len.float().unsqueeze(1)).item()
    pos_cor = torch.sum((pos == pos_preds).float() / (caption_len - 1).float().unsqueeze(1)).item()

    return [word_loss, cor, pos_cor, pos_loss]


if __name__ == "__main__":

    args = get_args()
    t = trainer(args)
    t.loop()
