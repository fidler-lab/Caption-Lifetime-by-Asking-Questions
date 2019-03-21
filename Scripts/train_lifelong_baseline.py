import torch
from time import time
import numpy as np
import random

import Utils.util as util
from Losses.metrics import mixed_reward
from Losses.loss import masked_PG
from util import LLBaseTrainer, get_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
random.seed(1)

class MuteTrainer(LLBaseTrainer):
    def __init__(self, args):
        super(MuteTrainer, self).__init__(args)

    def load_previous_chunk_best_models(self):
        self.reset_captioner(round=self.chunk)

    def do_iteration(self, image, refs, ref_lens, index):

        # self.set_seed(random.randint(1, 10000))

        # 1. Caption completely
        # sampled caption
        r = self.captioner.sample(image, greedy=False, max_seq_len=self.opt.c_max_sentence_len+1,
                                  temperature=self.opt.temperature)

        caption, log_probs, mask = r.caption, r.log_prob, r.mask

        # greedy caption
        gr = self.captioner.sample(image, greedy=True, max_seq_len=self.opt.c_max_sentence_len+1)
        caption_greedy, mask_greedy = gr.caption, gr.mask

        # 2. Compute reward for captions
        rwd = mixed_reward(caption, torch.sum(mask, dim=1), refs, ref_lens, self.scorers, self.c_i2w)
        rwd_greedy = mixed_reward(caption_greedy, torch.sum(mask_greedy, dim=1), refs, ref_lens, self.scorers, self.c_i2w)

        reward_delta = torch.from_numpy(rwd-rwd_greedy).to(device).type(torch.float)
        mask_copy = mask.clone()

        length = torch.clamp(torch.sum(mask != 0, dim=1).long().unsqueeze(1), max=self.opt.c_max_sentence_len)
        mask.scatter_(1, length, 1)  # increase mask dim by 1 to penalize for <eos>

        loss = masked_PG(reward_delta.unsqueeze(1), log_probs, mask)

        # 3. Collect data, we make use of the DataCollector class
        captions = [caption_greedy, caption, caption]
        cap_masks = [mask_greedy, mask_copy, mask_copy]

        self.data_collector.collect_batch(index.clone(), captions, cap_masks, [rwd_greedy, rwd, rwd])

        return loss.item()

    def train(self):

        print_loss, tic = 0, time()

        for i, sample in enumerate(self.train_loader):

            image, refs = [x.to(device) for x in [sample[0], sample[4]]]
            ref_lens, img_path, index = sample[5], sample[7], sample[8]

            batch_loss = self.do_iteration(image, refs, ref_lens, index)

            print_loss += batch_loss

            info = {
                'collect/loss': print_loss/self.opt.print_every,
                'collect/time': (time() - tic)/self.opt.print_every  # total time so far for this epoch
            }
            util.step_logging(self.logger, info, self.collection_steps)

            if self.collection_steps % self.opt.print_every == 0:

                steps_per_epoch = len(self.train_loader)
                self.std_logger.info(
                    "Baseline - Chunk {} Epoch {}, {}/{}| Loss: {} | Time per batch: {} |"
                    " Epoch remaining time (HH:MM:SS) {} | Elapsed time {}"
                        .format(self.chunk+1, self.collection_epoch, i, steps_per_epoch, info['collect/loss'],
                                info['collect/time'], util.time_remaining(steps_per_epoch - i, info['collect/time']),
                                util.time_elapsed(self.start_time, time())))

                print_loss, tic = 0, time()

            self.collection_steps += 1

        self.data_collector.process_collected_data()

    def loop_chunk(self, repochs):

        for epoch in range(repochs):
            self.train()

            info, distrs = self.data_collector.get_epoch_stats()
            info = {'mk-main/'+k:v for k,v in info.iteritems()}
            util.epoch_logging(self.logger, info, self.collection_epoch)
            util.distr_logging(self.logger, distrs, self.collection_epoch)
            self.data_collector.reset_epoch_counters()

            self.collection_epoch += 1


if __name__ == "__main__":

    args = get_args()
    t = MuteTrainer(args)
    t.loop_lifelong()
