import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

import re
import pickle
import logging
from time import strftime, gmtime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Bunch(object):
    def __init__(self, dict=None, **kwargs):
        if dict is not None:
            self.__dict__.update(dict)
        else:
            for name in kwargs:
                setattr(self, name, kwargs[name])


def update_scheduled_sampling(epoch, ss_start, ss_increase_every, ss_increase_prob, ss_max_prob, ss_init_value=0.0):
    frac = (epoch - ss_start) // ss_increase_every
    ss_prob = min(ss_init_value + ss_increase_prob * frac, ss_max_prob)
    return ss_prob


def update_lr(epoch, lr, lr_decay_start, lr_decay_every, lr_decay_rate):
    frac = (epoch - lr_decay_start) // lr_decay_every
    decay_factor = lr_decay_rate ** frac
    current_lr = lr * decay_factor
    return current_lr


def create_folder(path):
    if os.path.exists(path):
        print('Warning %s exists.'% (path))
    else:
        os.system('mkdir -p %s' % (path))


def time_remaining(steps_remaining, time_per_step):
    return strftime('%H:%M:%S', gmtime(steps_remaining * time_per_step))


def time_elapsed(start, end):
    return strftime('%H:%M:%S', gmtime(end - start))


def pad_sentence(sentence, max_len, pad_idx):
    l = len(sentence)
    pad_len = max(0, max_len - l)
    return sentence + [pad_idx] * pad_len


def to_np(x):
    if type(x) is np.ndarray:
        return x
    elif hasattr(x, 'data'):
        return x.data.cpu().numpy()  # variable
    else:
        return x.cpu().numpy()  # tensor


def clean_str(x):
    x = x.replace("'", "").replace("<unk>", "unk").replace(":", "")
    x = re.sub(r'(\S)(&)', r'\1', x)
    return x


def clean_pos(x):
    if '-LRB-' in x:
        x.remove('-LRB-')
    if '-RRB-' in x:
        x.remove('-RRB-')
    return x


def safe_dict_retrieval(c_i2w, special_symbol, x):
    if x in c_i2w:
        return c_i2w[x]
    else:
        return c_i2w[special_symbol]


def add_or_replace(l, i, x):
    if i >= len(l):
        l.append(x)
    else:
        l[i] = x


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def gradient_noise_and_clip(parameters, max_clip):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    nn.utils.clip_grad_norm(parameters, max_clip)


def weights_init(module):
    if isinstance(module, nn.Linear):
        init.xavier_normal_(module.weight)
        init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.Conv1d):
        init.normal_(module.weight, mean=0, std=0.01)
    elif isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0.01)
            elif 'weight' in name:
                init.xavier_normal_(param)


# LOGGING

def get_std_logger(name, log_file):

    logger = logging.getLogger(name)
    if len(logger.parent.handlers) > 0:  # weird i have to do this hack
        logger.parent.handlers = []
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def epoch_logging(logger, info, iteration):
    for tag, value in info.items():
        logger.scalar_summary("epochlogging/"+tag, value, iteration)


def round_logging(logger, info, iteration):
    for tag, value in info.items():
        logger.scalar_summary("roundlogging/"+tag, value, iteration)


def distr_logging(logger, info, iteration):
    for tag, distr in info.items():
        logger.histo_summary(tag, distr, iteration)


def parameter_logging(logger, model, iteration):
    # Log values and gradients of the parameters (histogram)
    for tag, value in model.named_parameters():
        if 'cnn.fc' in tag or 'cnn' not in tag:
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, to_np(value), iteration)
            if value.grad is not None:
                logger.histo_summary(tag + '/grad', to_np(value.grad), iteration)


def step_logging(logger, info, iteration):
    for tag, value in info.items():
        if 'hist' in tag:
            logger.histo_summary("steplogging/"+tag, to_np(value), iteration)
        else:
            logger.scalar_summary("steplogging/"+tag, value, iteration)


def log_avg_grads(logger, model, iteration, name=""):
    for tag, value in model.named_parameters():
        if 'weight' in tag and value.grad is not None and value.grad.numel() > 1:  # you get divide by 0 for torch.std if it's size 1
            tag = name + "grad/" + tag.replace('.', '/')
            logger.scalar_summary(tag + ' mean', to_np(torch.norm(value.grad)), iteration)
            logger.scalar_summary(tag + ' std', to_np(torch.std(value.grad)), iteration)

# QGen helpers
def qgen_getprobs_vqa(model, image, question, captions, mask, config):
    pad_symbol = config.q_vocab_size

    mask_inv = mask == 0
    question = question * mask.long() + (pad_symbol * torch.ones(mask.size(), dtype=long)).to(device) * mask_inv.long()

    result = model(image, question, captions)
    probs = result.probs

    return probs

def query_vqa(model, image, question, captions, answer, mask, config):

    pad_symbol = config.q_vocab_size

    mask_inv = mask == 0
    question = question * mask.long() + (pad_symbol * torch.ones(mask.size(), dtype=long)).to(device) * mask_inv.long()

    result = model(image, question, captions)
    probs = result.probs

    _, output_max_index = torch.max(probs, 1)

    num_correct = (answer == output_max_index).float().sum().item()

    return num_correct

def qgen_communicate_vqa_topk(model, image, question, captions, answer, mask, config):

    probs = qgen_getprobs_vqa(model, image, question, captions, mask, config)

    _, output_topk = torch.topk(probs, k=10, dim=1)
    output_top1 = output_topk[:, :1]
    output_top3 = output_topk[:, :3]
    output_top5 = output_topk[:, :5]
    output_top10 = output_topk[:, :10]

    top1_corr = torch.sum(((answer.unsqueeze(1) - output_top1) == 0).long(), dim=1).float().sum().data[0]
    top3_corr = torch.sum(((answer.unsqueeze(1) - output_top3) == 0).long(), dim=1).float().sum().data[0]
    top5_corr = torch.sum(((answer.unsqueeze(1) - output_top5) == 0).long(), dim=1).float().sum().data[0]
    top10_corr = torch.sum(((answer.unsqueeze(1) - output_top10) == 0).long(), dim=1).float().sum().data[0]

    probs = torch.gather(probs, dim=1, index=answer.unsqueeze(1))
    reward = torch.log(probs)
    reward = torch.clamp(reward, config.reward_logprob_clamp) + \
             config.reward_cor_bonus * ((answer == output_top1.squeeze(1)).float()).unsqueeze(
        1) + config.reward_bias

    return reward, top1_corr, top3_corr, top5_corr, top10_corr


def init_state(batch_size, vec_ones, hid_zeros):
    previous_word = vec_ones[:batch_size].clone()
    hidden = hid_zeros[:, :batch_size, :].clone()

    return hidden, previous_word

def init_vars(config, batch_size):
    words = to_var(torch.zeros(batch_size, config.max_sentence_len+1).long())
    mask = to_var(torch.zeros(batch_size, config.max_sentence_len+1).byte())
    decisions = to_var(torch.zeros(batch_size, config.max_sentence_len+1).byte())
    inter_rewards = to_var(torch.zeros(batch_size, config.max_sentence_len+1), requires_grad=False)
    hiddens = to_var(torch.zeros(batch_size, config.max_sentence_len + 1, config.c_rnn_size))

    return words, mask, decisions, inter_rewards, hiddens

def init_entropy():
    pos_entropies = [0, 0]
    decision_entropies = [0, 0]
    answer_entropies = [0, 0]
    ques_entropies = [0, 0]
    percent_ask = [0, 0]

    return Bunch(pe=pos_entropies, de=decision_entropies, ae=answer_entropies, qe=ques_entropies, pa=percent_ask)

def update_entropy(entropies, word_entropy, pos_entropy, d_entropy, a_entropy, q_entropy, q_mask, decision, global_unfinished, ask_worthy):
    word_entropies = entropies.we
    pos_entropies = entropies.pe
    decision_entropies = entropies.de
    answer_entropies = entropies.ae
    quest_entropies = entropies.qe
    percent_ask = entropies.pa

    global_unfinished = global_unfinished.float()
    unfinished_and_ask = global_unfinished * decision.float()
    d1 = global_unfinished.sum().data[0]
    d2 = unfinished_and_ask.sum().data[0]
    d3 = (global_unfinished * ask_worthy.float()).sum().data[0]

    word_entropies[0] += (word_entropy * global_unfinished).sum().data[0]
    word_entropies[1] += d1
    pos_entropies[0] += (pos_entropy * global_unfinished).sum().data[0]
    pos_entropies[1] += d1
    decision_entropies[0] += (d_entropy * global_unfinished).sum().data[0]
    decision_entropies[1] += d1
    answer_entropies[0] += (a_entropy * unfinished_and_ask).sum().data[0]
    answer_entropies[1] += d2
    q_mask = q_mask.float()
    q_entropy = torch.sum(q_entropy * q_mask, dim=1) / q_mask.sum(dim=1)
    quest_entropies[0] += (q_entropy * unfinished_and_ask).sum().data[0]
    quest_entropies[1] += d2
    percent_ask[0] += d2
    percent_ask[1] += d3

def update_entropy_(entropies, pos_probs, masked_prob, ans_prob, q_entropy, q_mask, will_ask, cap_mask, ask_worthy):
    pos_entropies = entropies.pe
    decision_entropies = entropies.de
    answer_entropies = entropies.ae
    quest_entropies = entropies.qe
    percent_ask = entropies.pa

    masked_prob = masked_prob.clone().detach()

    cap_mask = cap_mask.float()
    will_ask = will_ask.float()
    q_mask = q_mask.float()
    d1 = cap_mask.sum().data[0]
    d2 = will_ask.sum().data[0]

    pos_entropies[0] += ((-pos_probs*torch.log(pos_probs)).sum(dim=2) * cap_mask).sum().data[0]
    pos_entropies[1] += d1
    d_lp = torch.log(masked_prob)
    d_lp[masked_prob == 0] = 0
    decision_entropies[0] += (-masked_prob*d_lp).sum().data[0]
    decision_entropies[1] += (masked_prob > 0).float().sum().data[0]
    ans_prob = torch.topk(ans_prob, 10, dim=1)[0]
    (-ans_prob * torch.log(ans_prob)).sum(dim=1)
    answer_entropies[0] += ((-ans_prob * torch.log(ans_prob)).sum(dim=1) * will_ask).sum().data[0]
    answer_entropies[1] += d2
    quest_entropies[0] += ((q_entropy * q_mask).sum(dim=1) * will_ask).sum().data[0]
    quest_entropies[1] += d2
    percent_ask[0] += d2
    percent_ask[1] += will_ask.size(0)

def safe_divide(entropies):
    pe = entropies.pe[0] / entropies.pe[1] if entropies.pe[1] != 0 else 0
    de = entropies.de[0] / entropies.de[1] if entropies.de[1] != 0 else 0
    ae = entropies.ae[0] / entropies.ae[1] if entropies.ae[1] != 0 else 0
    qe = entropies.qe[0] / entropies.qe[1] if entropies.qe[1] != 0 else 0
    pa = entropies.pa[0] / entropies.pa[1] if entropies.pa[1] != 0 else 0

    return Bunch(pe=pe, de=de, ae=ae, qe=qe, pa=pa)

def prepare_entropies(entropies):
    info = {
        '-entropies/average pos entropy': entropies.pe,
        '-entropies/average question entropy': entropies.qe,
        '-entropies/average decision entropy': entropies.de,
        '-entropies/average answer entropy': entropies.ae,
        '/percent ask': entropies.pa
    }
    return info

def prepare_reward_stats(reward):
    info = {
        '/reward/average reward': torch.mean(reward).data[0],
        '/reward/reward std': torch.std(reward).data[0]
    }
    return info

def prepare_step_log(entropies, reward):
    info = {}
    info.update(prepare_entropies(entropies))
    info.update(prepare_reward_stats(reward))

    return info

def prepare_evalstep_log(entropies):
    info = {
        '-entropies/average pos entropy': entropies.pe,
        '-entropies/average question entropy': entropies.qe,
        '-entropies/average decision entropy': entropies.de,
        '-entropies/average answer entropy': entropies.ae,
        '/percent ask': entropies.pa
    }
    return info

def idx2str(idx2str_dictionary, inds):
    words = []
    for ind in inds:
        if ind < 3:
            words.append(idx2str_dictionary[ind].replace("<", "[").replace(">", "]"))
        elif ind == len(idx2str_dictionary):
            words.append('[pad]')
        else:
            words.append(idx2str_dictionary[ind])
    return words


def idx2pos(p_i2w, inds):
    pos = []
    for i in inds:
        if i == len(p_i2w):
            pos.append("[pad]")
        elif i == len(p_i2w)-1:
            pos.append("OTR")
        else:
            pos.append(p_i2w[i])
    return pos
