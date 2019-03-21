import os
import json
import argparse
from time import time
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dataset_vqa import VQADataset
from Models.attention_vqa import AttentionVQA
from Utils.tensorboardlogger import TensorboardLogger
import Utils.util as util
from Utils.util import Bunch
from Utils.visualizer import VQAVisualizer


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
        self.visualizer = VQAVisualizer(os.path.join(self.result_path, "{}.html".format(self.opt.run_name)),
                                        os.path.join(self.opt.exp_dir, "Utils/css/vqa.css"))

        self.get_data_loaders()

        # MODEL
        self.model = AttentionVQA(self.opt).to(device)
        self.model.apply(util.weights_init)

        # OPTIMIZER
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opt.lr)
        self.loss_function = nn.BCEWithLogitsLoss()

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

        train_data = VQADataset(self.opt, self.opt.train_file)
        val_data = VQADataset(self.opt, self.opt.val_file)

        self.train_loader = DataLoader(train_data, batch_size=self.opt.batch_size, num_workers=1, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=self.opt.batch_size, num_workers=1, shuffle=False)

        self.opt.c_vocab_size, self.opt.q_vocab_size, self.opt.a_vocab_size \
            = train_data.c_vocab_size, train_data.q_vocab_size, train_data.a_vocab_size

        self.std_logger.info("Size of question vocab: {} \n Size of answer vocab: {} \n Size of caption vocab: {} \n Max length of questions: {}"
                             .format(self.opt.q_vocab_size, self.opt.a_vocab_size, self.opt.c_vocab_size, train_data.max_q_len))

    def loop(self):
        for epoch in range(self.epoch, self.opt.max_epochs):
            self.epoch = epoch

            self.train(epoch)

    def train(self, epoch):

        print("Training")

        print_loss, tic = 0, time()

        for i, sample in enumerate(self.train_loader):

            image, question, question_len, answer, captions = sample[:-3]
            image, question, captions, answer = [x.to(device) for x in [image, question, captions, answer]]

            self.optimizer.zero_grad()

            # Forward pass
            result = self.model(image, question, captions)
            logits = result.logits

            # Get loss
            loss = self.loss_function(logits, answer)  # answer is coming in as double for some reason

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

        model_score = self.evaluate(epoch + 1)
        self.save_checkpoint(epoch, model_score)

    def evaluate(self, epoch):

        samples, [val_loss, val_acc] = self.validate()
        # TODO: implement VQA2.0 scoring score = eval_vqa(samples)

        info = {
            'val loss': val_loss,
            'val accuracy': val_acc
        }

        model_score = val_acc

        self.std_logger.info(str(info))
        util.epoch_logging(self.logger, info, epoch)

        return model_score

    def validate(self):

        print("Validating")

        self.model.eval()
        loss, correct = 0.0, 0.0
        samples = []

        for step, sample in enumerate(self.val_loader):

            image, question, question_len, answer, captions, cap_lens, img_path, que_id = sample
            image, question, answer, captions = [x.to(device) for x in [image, question, answer, captions]]

            # Forward pass
            result = self.model(image, question, captions)
            probs, logits = result.probs, result.logits

            loss += self.loss_function(logits, answer).item()

            # Compute top answer accuracy
            _, prediction_max_index = torch.max(probs, 1)
            _, answer_max_index = torch.max(answer, 1)

            correct += (answer_max_index == prediction_max_index).float().sum().item()

            a_i2w = self.val_loader.dataset.a_i2w
            q_i2w = self.val_loader.dataset.q_i2w
            c_i2w = self.val_loader.dataset.c_i2w

            # Append prediction to VQA2.0 validation
            for i in range(image.size(0)):
                samples.append({'question_id': que_id[i].item(), 'answer': a_i2w[prediction_max_index[i].item()]})

            # write image and Q&A to html file to visualize training progress
            if step % self.opt.visualize_every == 0:

                # Show the top 3 predicted answers
                pros, ans = torch.topk(probs, k=3, dim=1)
                captions, cap_lens, question, question_len, pros, ans =\
                    [util.to_np(x) for x in [captions, cap_lens, question, question_len, pros, ans]]

                for i in range(image.size(0)/2):

                    # Show question
                    que_arr = util.idx2str(q_i2w, (question[i])[:question_len[i]])

                    entry = {
                        'img': img_path[i],
                        'epoch': self.epoch,
                        'question': ' '.join(que_arr),
                        'gt_ans': a_i2w[answer_max_index[i].item()],
                        'predictions': [[p, a_i2w[a]] for p, a in zip(pros[i], ans[i])],
                        'refs': [
                            ' '.join(util.idx2str(c_i2w, (captions[i, j])[:cap_lens[i, j]])) for j in range(3)
                        ]
                    }
                    self.visualizer.add_entry(entry)

        self.visualizer.update_html()
        # Reset
        self.model.train()

        return samples, [x / len(self.val_loader.dataset) for x in [loss, correct]]


if __name__ == "__main__":

    args = get_args()
    t = trainer(args)
    t.loop()
