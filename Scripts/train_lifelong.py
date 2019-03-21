import os
import torch
from time import time
from argparse import Namespace
import numpy as np
import random

import Utils.util as util
from Models.decision_maker import DecisionMaker

from eval.coco_eval import language_scores
from Models.attention_captioner import AttentionCaptioner
from Models.attention_questioner import QuestionGenerator
from Models.attention_vqa import AttentionVQA
from Losses.metrics import mixed_reward, linear_reward_weighting
from Losses.loss import masked_PG
from util import LLBaseTrainer
from Utils.visualizer import LLVisualizer
from util import get_args, load_model, get_rollout_replace_stats, replace_word_in_caption,\
    choose_better_caption, masked_softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
random.seed(1)

class QuestionAskingTrainer(LLBaseTrainer):
    def __init__(self, args):
        super(QuestionAskingTrainer, self).__init__(args)

        # initialize part-of-speech parser
        self.eval_steps = 0

        # initialize models
        self.qgen = load_model(self.opt.quegen_path, QuestionGenerator)
        self.vqa = load_model(self.opt.vqa_path, AttentionVQA)
        self.fixed_caption_encoder = load_model(self.opt.cap_path, AttentionCaptioner)
        self.dmaker = DecisionMaker(self.opt).to(device)

        self.dmaker.apply(util.weights_init)
        self.dmaker.pos_embedding.weight.data.copy_(self.captioner.decoder.pos_embedding.weight.data)
        self.dmaker.caption_embedding.weight.data.copy_(self.captioner.caption_embedding.weight.data)

        self.d_optimizer = torch.optim.Adam(
            [
                {'params': filter(lambda p: p.requires_grad, self.dmaker.parameters()), 'lr': self.opt.d_lr,
                 'mod_name': 'dec'}
            ])

        self.fixed_caption_encoder.eval()
        self.vqa.eval()

        self.init_dummy_tensors()

        # Logging
        self.trainLLvisualizer = LLVisualizer(os.path.join(self.result_path, "{}_lltrain.html".format(self.opt.run_name)),
                                              os.path.join(self.opt.exp_dir, "Utils/css/lifelong.css"))
        self.valLLvisualizer = LLVisualizer(os.path.join(self.result_path, "{}_llval.html".format(self.opt.run_name)),
                                              os.path.join(self.opt.exp_dir, "Utils/css/lifelong.css"))

    def init_dummy_tensors(self):

        # Put dummy tensors on GPU to speed up training
        self.range_vector = torch.arange(self.opt.batch_size, dtype=torch.long, device=device)
        self.range_matrix = torch.arange(self.opt.c_max_sentence_len + 1, dtype=torch.long, device=device)\
            .repeat(self.opt.batch_size, 1)
        self.ones_vector = torch.ones(self.opt.batch_size, dtype=torch.long, device=device, requires_grad=False)
        self.ones_matrix = torch.ones([self.opt.batch_size, self.opt.q_max_sentence_len+1],
                                      dtype=torch.long, device=device, requires_grad=False)
        self.zeros_matrix = torch.zeros([self.opt.batch_size, self.opt.c_max_sentence_len + 1],
                                        dtype=torch.long, device=device, requires_grad=False)
        self.zeros_tensor = torch.zeros([self.opt.batch_size, self.opt.c_max_sentence_len + 1, self.opt.dm_rnn_size],
                                        dtype=torch.float, device=device, requires_grad=False)
        self.zeros_tensor2 = torch.zeros([self.opt.rnn_layers, self.opt.batch_size, self.opt.rnn_size],
                                         dtype=torch.float, device=device, requires_grad=False)

    def load_previous_chunk_best_models(self):
        # load decision maker model
        self.std_logger.info("Loading decision maker from {}.".format(self.decision_maker_model_files[self.chunk]))
        save_state = torch.load(self.decision_maker_model_files[self.chunk], map_location=lambda storage, loc: storage)
        self.dmaker.load_state_dict(save_state['state_dict'])
        # load caption model
        self.reset_captioner(round=self.chunk)

    def sample_decision(self, masked_prob, caption_mask, greedy=False):
        batch_size = masked_prob.size(0)
        zeros_mask = self.zeros_matrix[:batch_size].clone()

        if greedy:
            masked_prob_copy = masked_prob.clone().detach()
            val, idx = torch.max(masked_prob_copy, 1)
            val, idx = val.unsqueeze(1), idx.unsqueeze(1)
        else:
            idx = torch.multinomial(masked_prob, 1)
            val = masked_prob.gather(1, idx)

        # edge-case: don't ask if probabilities are all 0s
        length = torch.clamp(torch.sum(caption_mask != 0, dim=1).long().unsqueeze(1), max=self.opt.c_max_sentence_len)
        ask_flag = (val != 0) * (val > self.opt.question_asking_threshold) * (idx != length)
        ask_mask = zeros_mask.scatter(1, idx, ask_flag.long()).detach()

        return [x.squeeze() for x in [val, idx, ask_flag, ask_mask]]

    def ask_question(self, image, caption, refs, pos_probs, h, att, idx, q_greedy, temperature=1.0):
        batch_size = pos_probs.size(0)

        # index features by the decision maker time step
        pos_probs = pos_probs[self.range_vector[:batch_size], idx].detach()
        h = h[self.range_vector[:batch_size], idx].detach()
        att = att[self.range_vector[:batch_size], idx].detach()

        # decision maker index vector
        q_idx_vec = self.zeros_tensor.clone()[:batch_size]
        q_idx_vec[self.range_vector[:batch_size], idx, :] = 1.0

        # query question generator
        result = self.qgen.sample(image, caption, pos_probs, h, att, q_idx_vec, greedy=q_greedy,
                                  max_seq_len=self.opt.q_max_sentence_len+1, temperature=temperature)
        question, q_logprob, q_mask = result.question, result.log_prob, result.mask
        question = self.pad_question(question, q_mask)  # replace 0's with

        # ask expert question and get answer
        result = self.vqa(image, question, refs)
        ans_prob = result.probs
        answers = torch.max(ans_prob, dim=1)[1]
        answer_mask = self.zeros_matrix[:batch_size].clone().scatter(1, idx.unsqueeze(1), answers.unsqueeze(1)).detach()

        return answers, answer_mask, Namespace(pos_prob=pos_probs, att=att, q_logprob=q_logprob, q_mask=q_mask,
                                               question=question, ans_prob=ans_prob)

    def caption_with_teacher_answer(self, image, ask_mask, answer_mask, greedy, temperature=1.0):

        hidden, previous_word = util.init_state(image.size(0), self.ones_vector, self.zeros_tensor2)
        self.set_seed(self.opt.seed)
        return self.captioner.sample_with_teacher_answer(image, ask_mask, answer_mask, hidden, previous_word,
                                                         self.opt.c_max_sentence_len + 1, greedy, temperature)

    def do_iteration(self, image, refs, ref_lens, index, img_path):
        self.d_optimizer.zero_grad()

        batch_size = image.size(0)

        caps, cmasks, qs, qlps, qmasks, aps, pps, atts, cps, dps = [], [], [], [], [], [], [], [], [], []

        # 1. Caption completely
        self.set_seed(self.opt.seed)

        r = self.captioner.sample(image, greedy=self.opt.cap_greedy, max_seq_len=self.opt.c_max_sentence_len+1,
                                  temperature=self.opt.temperature)

        caption, cap_probs, cap_mask, pos_probs, att, topk_words, attended_img \
            = r.caption, r.prob, r.mask, r.pos_prob, r.attention.squeeze(), r.topk, r.atdimg

        # Don't backprop through captioner
        caption, cap_probs, cap_mask, pos_probs, attended_img \
            = [x.detach() for x in [caption, cap_probs, cap_mask, pos_probs, attended_img]]

        cap_len = cap_mask.long().sum(dim=1)
        caps.append(caption)
        cmasks.append(cap_mask)

        caption = self.pad_caption(caption, cap_len)

        # get the hidden state context
        source = torch.cat([self.ones_vector[:batch_size].unsqueeze(1), caption[:, :-1]], dim=1)
        r = self.fixed_caption_encoder(image, source, gt_pos=None, ss=False)
        h = r.hidden.detach()
        topk_words = [[y.detach() for y in x] for x in topk_words]

        # 2. Identify the best time to ask a question, excluding ended sentences, baseline against the greedy decision
        logit, valid_pos_mask = self.dmaker(h, attended_img, caption, cap_len,
                                        pos_probs, topk_words, self.captioner.caption_embedding.weight.data)
        masked_prob = masked_softmax(logit, cap_mask, valid_pos_mask,
                                          self.opt.dm_temperature, max_len=self.opt.c_max_sentence_len)

        dm_prob, ask_idx, ask_flag, ask_mask = self.sample_decision(masked_prob, cap_mask, greedy=False)
        _, ask_idx_greedy, ask_flag_greedy, ask_mask_greedy = self.sample_decision(masked_prob, cap_mask, greedy=True)

        dps.append(dm_prob.unsqueeze(1))

        # 3. Ask the teacher a question and get the answer
        ans, ans_mask, r = self.ask_question(image, caption, refs, pos_probs, h, att, ask_idx,
                                             q_greedy=self.opt.q_greedy, temperature=self.opt.temperature)
        ans_greedy, ans_mask_greedy, rg = self.ask_question(image, caption, refs, pos_probs, h, att, ask_idx_greedy,
                                                q_greedy=self.opt.q_greedy, temperature=self.opt.temperature)

        # logging stuff
        cps.append(cap_probs[self.range_vector[:batch_size], ask_idx]); cps.append(cap_probs[self.range_vector[:batch_size], ask_idx_greedy])
        pps.append(r.pos_prob[0]); pps.append(rg.pos_prob[0])
        atts.append(r.att[0]); atts.append(rg.att[0])
        qlps.append(r.q_logprob.unsqueeze(1)); qlps.append(rg.q_logprob.unsqueeze(1))
        qmasks.append(r.q_mask); qmasks.append(rg.q_mask)
        qs.append(r.question.detach()); qs.append(rg.question.detach())
        aps.append(r.ans_prob.detach()); aps.append(rg.ans_prob.detach())

        # 4. Compute new captions based on teacher's answer
        # rollout caption
        r = self.caption_with_teacher_answer(image, ask_mask, ans_mask,
                                             greedy=self.opt.cap_greedy, temperature=self.opt.temperature)
        rg = self.caption_with_teacher_answer(image, ask_mask_greedy, ans_mask_greedy,
                                              greedy=self.opt.cap_greedy, temperature=self.opt.temperature)

        rollout, rollout_mask, rollout_greedy, rollout_mask_greedy = [x.detach() for x in
                                                                      [r.caption, r.cap_mask, rg.caption, rg.cap_mask]]

        # replace caption
        replace = replace_word_in_caption(caps[0], ans, ask_idx, ask_flag)
        replace_greedy = replace_word_in_caption(caps[0], ans_greedy, ask_idx_greedy, ask_flag_greedy)

        # 5. Compute reward for captions
        base_rwd = mixed_reward(caps[0], torch.sum(cmasks[0], dim=1), refs, ref_lens, self.scorers, self.c_i2w)
        rollout_rwd = mixed_reward(rollout, torch.sum(rollout_mask, dim=1), refs, ref_lens, self.scorers, self.c_i2w)
        rollout_greedy_rwd = mixed_reward(rollout_greedy, torch.sum(rollout_mask_greedy, dim=1), refs, ref_lens,
                                          self.scorers, self.c_i2w)

        replace_rwd = mixed_reward(replace, torch.sum(cmasks[0], dim=1), refs, ref_lens, self.scorers, self.c_i2w)
        replace_greedy_rwd = mixed_reward(replace_greedy, torch.sum(cmasks[0], dim=1), refs, ref_lens,
                                          self.scorers, self.c_i2w)

        rwd = np.maximum(replace_rwd, rollout_rwd)
        rwd_greedy = np.maximum(replace_greedy_rwd, rollout_greedy_rwd)

        best_cap, best_cap_mask = choose_better_caption(
            replace_rwd, replace, cmasks[0], rollout_rwd, rollout, rollout_mask)
        best_cap_greedy, best_cap_greedy_mask = choose_better_caption(
            replace_greedy_rwd, replace_greedy, cmasks[0], rollout_greedy_rwd, rollout_greedy, rollout_mask_greedy)

        # some statistics on whether rollout or single-word-replace is better
        stat_rero, stat_rore, stat_reall, stat_roall = get_rollout_replace_stats(replace_rwd, rollout_rwd, base_rwd)
        caps.append(best_cap); cmasks.append(best_cap_mask)
        caps.append(best_cap_greedy); cmasks.append(best_cap_greedy_mask)

        # Backwards pass to train decision maker with policy gradient loss
        reward_delta = torch.from_numpy(rwd-rwd_greedy).type(torch.float).to(device)
        reward_delta = reward_delta - self.opt.ask_penalty * ask_flag.float()

        loss = masked_PG(reward_delta.detach(), torch.log(dps[0]).squeeze(), ask_flag.detach())
        loss.backward()

        self.d_optimizer.step()

        # also save the question asked and answer, and top-k predictions from captioner
        answers = [torch.max(x, dim=1)[1] for x in aps]
        topwords = [torch.topk(x, 20)[1] for x in cps]
        question_lens = [x.sum(dim=1) for x in qmasks]
        self.data_collector.collect_batch(index.clone(), caps, cmasks, [base_rwd, rwd, rwd_greedy], answers, qs, question_lens, topwords)

        # Logging
        if self.collection_steps % self.opt.print_every == 0:

            c_i2w = self.val_loader.dataset.c_i2w
            p_i2w = self.val_loader.dataset.p_i2w
            q_i2w = self.val_loader.dataset.q_i2w
            a_i2w = self.val_loader.dataset.a_i2w

            caption, rollout, replace, cap_len, rollout_len, dec_probs, question, q_logprob, q_len,\
            flag, raw_idx, refs, ref_lens = \
                [util.to_np(x) for x in [caps[0], rollout, replace, cmasks[0].long().sum(dim=1),
                                         rollout_mask.long().sum(dim=1), masked_prob, qs[0], qlps[0],
                                         qmasks[0].long().sum(dim=1), ask_flag.long(),
                                         ask_idx, refs, ref_lens]]
            pos_probs, ans_probs, cap_probs = pps[0], aps[0], cps[0]

            for i in range(image.size(0)):

                top_pos = torch.topk(pos_probs, 3)[1]
                top_ans = torch.topk(ans_probs[i], 3)[1]
                top_cap = torch.topk(cap_probs[i], 5)[1]

                word = c_i2w[caption[i][raw_idx[i]]] if raw_idx[i]<len(caption[i]) else 'Nan'

                entry = {
                    'img': img_path[i],
                    'epoch': self.collection_epoch,
                    'caption': ' '.join(util.idx2str(c_i2w, (caption[i])[:cap_len[i]]))
                               + " | Reward: {:.2f}".format(base_rwd[i]),
                    'qaskprobs': ' '.join(["{:.2f}".format(x) if x > 0.0 else "_" for x in dec_probs[i]]),
                    'rollout_caption': ' '.join(util.idx2str(c_i2w, (rollout[i])[:rollout_len[i]]))
                                       + " | Reward: {:.2f}".format(rollout_rwd[i]),
                    'replace_caption': ' '.join(util.idx2str(c_i2w, (replace[i])[:cap_len[i]]))
                                       + " | Reward: {:.2f}".format(replace_rwd[i]),
                    'index': raw_idx[i],
                    'flag': bool(flag[i]),
                    'word': word,
                    'pos': ' | '.join([p_i2w[x.item()] if x.item() in p_i2w else p_i2w[18] for x in top_pos]),
                    'question': ' '.join(util.idx2str(q_i2w, (question[i])[:q_len[i]]))
                                + " | logprob: {}".format(q_logprob[i, :q_len[i]].sum()),
                    'answers': ' | '.join([a_i2w[x.item()] for x in top_ans]),
                    'words': ' | '.join([c_i2w[x.item()] for x in top_cap]),
                    'refs': [
                        ' '.join(util.idx2str(c_i2w, (refs[i, j])[:ref_lens[i, j]])) for j in range(3)
                    ]
                }

                self.trainLLvisualizer.add_entry(entry)

                info = {
                    'collect/question logprob': torch.mean(torch.cat(qlps, dim=1)).item(),
                    'collect/replace over rollout': float(stat_rero) / (batch_size),
                    'collect/rollout over replace': float(stat_rore) / (batch_size),
                    'collect/replace over all': float(stat_reall) / (batch_size),
                    'collect/rollout over all': float(stat_roall) / (batch_size),
                    'collect/sampled decision equals greedy decision': float((ask_idx == ask_idx_greedy).sum().item()) / (batch_size),
                    'collect/question asking frequency (percent)': float(ask_flag.sum().item())/ask_flag.size(0)
                        }
                util.step_logging(self.logger, info, self.collection_steps)

        return loss.item()

    def train(self):

        print_loss, tic = 0, time()

        for i, sample in enumerate(self.train_loader):

            image, refs = [x.to(device) for x in [sample[0], sample[4]]]
            ref_lens, img_path, index = sample[5], sample[7], sample[8]
            batch_loss = self.do_iteration(image, refs, ref_lens, index, img_path)

            print_loss += batch_loss

            info = {
                'collect/loss': print_loss/self.opt.print_every,
                'collect/time': (time() - tic)/self.opt.print_every  # total time so far for this epoch
            }
            util.step_logging(self.logger, info, self.collection_steps)

            if self.collection_steps % self.opt.print_every == 0:
                util.log_avg_grads(self.logger, self.dmaker, self.collection_steps, name="dec")
                steps_per_epoch = len(self.train_loader)
                self.std_logger.info(
                    "Chunk {} Epoch {}, {}/{}| Loss: {} | Time per batch: {} |"
                    " Epoch remaining time (HH:MM:SS) {} | Elapsed time {}"
                        .format(self.chunk+1, self.collection_epoch, i, steps_per_epoch, info['collect/loss'], info['collect/time'],
                                util.time_remaining(steps_per_epoch - i, info['collect/time']),
                                util.time_elapsed(self.start_time, time())))

                print_loss, tic = 0, time()

            self.collection_steps += 1

        self.trainLLvisualizer.update_html()
        self.data_collector.process_collected_data()

    def loop_chunk(self, repochs):

        if self.opt.cap_eval:
            self.captioner.eval()
        else:
            self.captioner.train()

        if self.opt.quegen_eval:
            self.qgen.eval()
        else:
            self.qgen.train()
        for epoch in range(repochs):
            self.std_logger.info("Training decision maker and collecting captions")

            self.train()

            info, distrs = self.data_collector.get_epoch_stats()
            info = {'mk-main/'+k:v for k,v in info.iteritems()}
            util.epoch_logging(self.logger, info, self.collection_epoch)
            util.distr_logging(self.logger, distrs, self.collection_epoch)
            self.data_collector.reset_epoch_counters()

            model_score = self.evaluate_decision_maker()
            self.save_decision_maker(epoch, model_score)

            self.collection_epoch += 1

    def save_decision_maker(self, epoch, model_score):

        save_state = {
            'epoch': epoch+1,
            'collection_steps': self.collection_steps,
            'collection_epoch': self.collection_epoch,
            'state_dict': self.dmaker.state_dict(),
            'optimizer': self.d_optimizer.state_dict(),
            'opt': self.opt,
            'best_score': self.best_d_score
        }

        if model_score > self.best_d_score:
            self.std_logger.info("New best decision maker score {} > {} previous score.".format(model_score, self.best_d_score))
            save_name = self.decision_maker_model_files[self.chunk]
            torch.save(save_state, save_name)
            self.best_d_score = model_score

    def evaluate_decision_maker(self):

        info = {}

        acc, scores = self.evaluate_with_questions()

        for i in range(len(acc)):
            info['mk-supp/val accuracy Q{}'.format(i)] = acc[i]
            info['mk-supp/CIDEr Q{}'.format(i)] = scores[i]['CIDEr'] * 100.0
            info['mk-supp/Bleu_4 Q{}'.format(i)] = scores[i]['Bleu_4'] * 100.0
            info['mk-supp/Bleu_3 Q{}'.format(i)] = scores[i]['Bleu_3'] * 100.0
            info['mk-supp/Bleu_2 Q{}'.format(i)] = scores[i]['Bleu_2'] * 100.0
            info['mk-supp/Bleu_1 Q{}'.format(i)] = scores[i]['Bleu_1'] * 100.0
            info['mk-supp/ROUGE_L Q{}'.format(i)] = scores[i]['ROUGE_L'] * 100.0
            info['mk-supp/METEOR Q{}'.format(i)] = scores[i]['METEOR'] * 100.0
            info['mk-main/weighted score Q{}'.format(i)] = \
                linear_reward_weighting(scores[i]['Bleu_1'], scores[i]['Bleu_2'], scores[i]['Bleu_3'],
                                        scores[i]['Bleu_4'], scores[i]['ROUGE_L'], scores[i]['METEOR'],
                                        scores[i]['CIDEr']) * 100.0

        model_score = linear_reward_weighting(scores[-1]['Bleu_1'], scores[-1]['Bleu_2'], scores[-1]['Bleu_3'],
                                      scores[-1]['Bleu_4'], scores[-1]['ROUGE_L'], scores[-1]['METEOR'],
                                      scores[-1]['CIDEr']) * 100.0

        self.std_logger.info("Round {} | Epoch {}: | Weighted score: {}".format(self.chunk+1, self.collection_epoch, model_score))

        util.epoch_logging(self.logger, info, self.collection_epoch)
        return model_score

    def evaluate_with_questions(self):

        self.std_logger.info("Validating decision maker")

        self.dmaker.eval()
        self.captioner.train()
        self.qgen.train()
        # if self.opt.cap_eval:
        #     self.captioner.eval()
        # else:
        #     self.captioner.train()
        #
        # if self.opt.quegen_eval:
        #     self.question_generator.eval()
        # else:
        #     self.question_generator.train()

        c_i2w = self.val_loader.dataset.c_i2w

        correct, pos_correct, eval_scores = [0.0, 0.0], [0.0, 0.0], []
        caption_predictions = [[], []]

        with torch.no_grad():

            for step, sample in enumerate(self.val_loader):

                image, refs, target, caption_len = [x.to(device) for x in [sample[0], sample[4], sample[2], sample[3]]]
                ref_lens, img_path, index, img_id = sample[5], sample[7], sample[8], sample[9]

                batch_size = image.size(0)

                caps, cmasks, poses, pps, qs, qlps, qmasks, aps, cps, atts = [], [], [], [], [], [],[], [], [], []

                # 1. Caption completely
                self.set_seed(self.opt.seed)

                r = self.captioner.sample(image, greedy=True, max_seq_len=self.opt.c_max_sentence_len+1)

                caption, cap_probs, cap_mask, pos_probs, att, topk_words, attended_img \
                    = r.caption, r.prob, r.mask, r.pos_prob, r.attention.squeeze(), r.topk, r.atdimg

                cap_len = cap_mask.long().sum(dim=1)
                caps.append(caption); cmasks.append(cap_mask); poses.append(pos_probs);

                caption = self.pad_caption(caption, cap_len)

                # get the hidden state context
                source = torch.cat([self.ones_vector[:batch_size].unsqueeze(1), caption[:, :-1]], dim=1)

                r = self.fixed_caption_encoder(image, source, gt_pos=None, ss=False)
                h = r.hidden

                # 2. Identify the best time to ask a question, excluding ended sentences
                logit, valid_pos_mask = self.dmaker(h, attended_img, caption, cap_len, pos_probs,
                                                topk_words, self.captioner.caption_embedding.weight.data)
                masked_prob = masked_softmax(logit, cap_mask, valid_pos_mask, max_len=self.opt.c_max_sentence_len)
                dm_prob, ask_idx, ask_flag, ask_mask = self.sample_decision(masked_prob, cap_mask, greedy=True)

                # 3. Ask the teacher a question and get the answer
                ans, ans_mask, r = self.ask_question(image, caption, refs, pos_probs, h, att, ask_idx, q_greedy=True)

                # logging
                cps.append(cap_probs[self.range_vector[:batch_size], ask_idx]); cps.append(cap_probs[self.range_vector[:batch_size], ask_idx])
                pps.append(r.pos_prob[0]); pps.append(r.pos_prob[0])
                atts.append(r.att[0]); atts.append(r.att[0])
                qlps.append(r.q_logprob.unsqueeze(1)); qlps.append(r.q_logprob.unsqueeze(1))
                qmasks.append(r.q_mask); qmasks.append(r.q_mask)
                qs.append(r.question); qs.append(r.question)
                aps.append(r.ans_prob); aps.append(r.ans_prob)

                # 4. Compute new captions based on teacher's answer
                # rollout caption
                r = self.caption_with_teacher_answer(image, ask_mask, ans_mask, greedy=True)

                poses.append(r.pos_prob)
                rollout = r.caption
                rollout_mask = r.cap_mask

                # replace caption
                replace = replace_word_in_caption(caps[0], ans, ask_idx, ask_flag)

                base_rwd = mixed_reward(caps[0], torch.sum(cmasks[0], dim=1), refs, ref_lens, self.scorers, self.c_i2w)
                rollout_rwd = mixed_reward(rollout, torch.sum(rollout_mask, dim=1), refs, ref_lens, self.scorers, self.c_i2w)
                replace_rwd = mixed_reward(replace, torch.sum(cmasks[0], dim=1), refs, ref_lens, self.scorers, self.c_i2w)

                caps.append(rollout)
                caps.append(replace)
                cmasks.append(rollout_mask)
                cmasks.append(cmasks[0])

                stat_rero, stat_rore, stat_reall, stat_roall = get_rollout_replace_stats(replace_rwd, rollout_rwd, base_rwd)

                best_cap, best_cap_mask = choose_better_caption(
                    replace_rwd, replace, cmasks[0], rollout_rwd, rollout, rollout_mask)

                caps = [caps[0], best_cap]
                cmasks = [cmasks[0], best_cap_mask]

                # Collect captions for coco evaluation
                img_id = util.to_np(img_id)

                for i in range(len(caps)):
                    words, lens, = util.to_np(caps[i]), util.to_np(cmasks[i].sum(dim=1))

                    for j in range(image.size(0)):
                        inds = words[j][:lens[j]]
                        caption = ""
                        for k, ind in enumerate(inds):
                            if k > 0:
                                caption = caption + ' '
                            caption = caption + c_i2w[ind]

                        pred = {
                            'image_id': img_id[j],
                            'caption': caption
                        }
                        caption_predictions[i].append(pred)

                for i in range(len(correct)):
                    predictions = caps[i] * cmasks[i].long()
                    correct[i] += ((target == predictions).float() / caption_len.float().unsqueeze(1)).sum().item()

                # Logging
                if step % self.opt.val_print_every == 0:

                    c_i2w = self.val_loader.dataset.c_i2w
                    p_i2w = self.val_loader.dataset.p_i2w
                    q_i2w = self.val_loader.dataset.q_i2w
                    a_i2w = self.val_loader.dataset.a_i2w

                    caption, rollout, replace, cap_len, rollout_len, dec_probs, question, q_logprob, q_len, \
                    flag, raw_idx, refs, ref_lens = \
                        [util.to_np(x) for x in [caps[0], rollout, replace, cmasks[0].long().sum(dim=1),
                                                 rollout_mask.long().sum(dim=1), masked_prob, qs[0], qlps[0],
                                                 qmasks[0].long().sum(dim=1), ask_flag.long(),
                                                 ask_idx, refs, ref_lens]]
                    pos_probs, ans_probs, cap_probs = pps[0], aps[0], cps[0]

                    for i in range(image.size(0)):
                        top_pos = torch.topk(pos_probs, 3)[1]
                        top_ans = torch.topk(ans_probs[i], 3)[1]
                        top_cap = torch.topk(cap_probs[i], 5)[1]

                        word = c_i2w[caption[i][raw_idx[i]]] if raw_idx[i] < len(caption[i]) else 'Nan'

                        entry = {
                            'img': img_path[i],
                            'epoch': self.collection_epoch,
                            'caption': ' '.join(util.idx2str(c_i2w, (caption[i])[:cap_len[i]]))
                                       + " | Reward: {:.2f}".format(base_rwd[i]),
                            'qaskprobs': ' '.join(["{:.2f}".format(x) if x > 0.0 else "_" for x in dec_probs[i]]),
                            'rollout_caption': ' '.join(util.idx2str(c_i2w, (rollout[i])[:rollout_len[i]]))
                                               + " | Reward: {:.2f}".format(rollout_rwd[i]),
                            'replace_caption': ' '.join(util.idx2str(c_i2w, (replace[i])[:cap_len[i]]))
                                               + " | Reward: {:.2f}".format(replace_rwd[i]),
                            'index': raw_idx[i],
                            'flag': bool(flag[i]),
                            'word': word,
                            'pos': ' | '.join([p_i2w[x.item()] if x.item() in p_i2w else p_i2w[18] for x in top_pos]),
                            'question': ' '.join(util.idx2str(q_i2w, (question[i])[:q_len[i]]))
                                        + " | logprob: {}".format(q_logprob[i, :q_len[i]].sum()),
                            'answers': ' | '.join([a_i2w[x.item()] for x in top_ans]),
                            'words': ' | '.join([c_i2w[x.item()] for x in top_cap]),
                            'refs': [
                                ' '.join(util.idx2str(c_i2w, (refs[i, j])[:ref_lens[i, j]])) for j in range(3)
                            ]
                        }

                        self.valLLvisualizer.add_entry(entry)

                    info = {'eval/replace over rollout': float(stat_rero)/(batch_size),
                            'eval/rollout over replace': float(stat_rore)/(batch_size),
                            'eval/replace over all': float(stat_reall) / (batch_size),
                            'eval/rollout over all': float(stat_roall) / (batch_size),
                            'eval/question asking frequency (percent)': float(ask_flag.sum().item())/ask_flag.size(0)
                            }
                    util.step_logging(self.logger, info, self.eval_steps)
                    self.eval_steps += 1

        self.valLLvisualizer.update_html()

        acc = [x / len(self.val_loader.dataset) for x in correct]

        for i in range(len(correct)):
            eval_scores.append(language_scores(caption_predictions[i], self.opt.run_name, self.result_path,
                                               annFile=self.opt.coco_annotation_file))

        self.dmaker.train()

        return acc, eval_scores


if __name__ == "__main__":

    args = get_args()
    t = QuestionAskingTrainer(args)
    t.loop_lifelong()
