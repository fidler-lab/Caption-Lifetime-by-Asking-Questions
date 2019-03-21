"""

Auto-eval metrics and computing the caption reward for lifelong learning.
"""

import sys
from Utils.util import to_np
from collections import OrderedDict
import numpy as np

sys.path.append("Dependencies/cider")
sys.path.append("Dependencies/coco-caption")
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pyciderevalcap.ciderD.ciderD import CiderD


def symbol_to_string(arr, arr_len, eos_symbol, c_i2w):
    max_valid_token = len(c_i2w)
    out = ''
    # if arr_len == len(arr) then the caption doesn't have the <eos> token
    if arr_len >= len(arr):
        for x in arr:
            if x < max_valid_token:
                out += c_i2w[x] + ' '
    else:
        for x in arr[: arr_len]:
            if x < max_valid_token:
                out += c_i2w[x] + ' '
        out += eos_symbol

    return out.strip()


def build_cap_ref_dicts(pred, pred_len, targ, targ_len, eos_symbol, c_i2w):
    pred, pred_len, ref, ref_len = [to_np(x).tolist() for x in [pred, pred_len, targ, targ_len]]

    gts = OrderedDict()
    for i in range(len(ref)):
        all_refs = []
        for j in range(len(ref[i])):
            all_refs.append(symbol_to_string(ref[i][j], ref_len[i][j], eos_symbol, c_i2w))
        gts[i] = all_refs

    res = [{'image_id': i, 'caption': [symbol_to_string(pred[i], pred_len[i], eos_symbol, c_i2w)]} for i in
           range(len(pred))]

    gts = {i: gts[i] for i in range(len(gts))}
    return gts, res

# the normalization is computed by making sure the new score sums to approximately 1
# the auto-eval scores are on average: rouge=0.5, meteor=0.25, cider=1.0, bleu4=0.25, bleu3=0.4, blue2=0.5, bleu1=0.7
# multiplying by the coefficients gives: 2*0.5 + 5* 0.25 + 1.5*1.0 + 1*0.25 + 1*0.4 + 0.5*0.5 + 0.5*0.7 = 5
def linear_reward_weighting(bleu1, bleu2, bleu3, bleu4, rouge, meteor, cider):
    c_b1 = 0.5/5.0
    c_b2 = 0.5/5.0
    c_b3 = 1.0/5.0
    c_b4 = 1.0/5.0
    c_r = 2.0/5.0
    c_m = 5.0/5.0
    c_c = 1.5/5.0
    return c_b1*bleu1 + c_b2*bleu2 + c_b3*bleu3 + c_b4*bleu4 + c_r*rouge + c_m*meteor + c_c*cider


def mixed_reward(pred, pred_len, targ, targ_len, scorers, c_i2w, eos_symbol='eos'):
    # prepare predicted captions and reference captions for scoring
    gts, res = build_cap_ref_dicts(pred, pred_len, targ, targ_len, eos_symbol, c_i2w)

    # score captions
    _, cider = scorers['cider'].compute_score(gts, res)

    # bleu, rouge, meteor require a different input format
    res_dict = {}
    for r in res:
        res_dict[r['image_id']] = r['caption']

    _, bleu = scorers['bleu'].compute_score(gts, res_dict)
    _, rouge = scorers['rouge'].compute_score(gts, res_dict)
    _, meteor = scorers['meteor'].compute_score(gts, res_dict)
    bleu1, bleu2, bleu3, bleu4 = [np.asarray(x) for x in bleu]

    # compute weighted average reward
    wt_r = linear_reward_weighting(bleu1, bleu2, bleu3, bleu4, rouge, np.asarray(meteor), cider)
    return wt_r


def get_scorers(cider_idx_path):
    return {
        'cider': CiderD(df=cider_idx_path),
        'bleu': Bleu(),
        'rouge': Rouge(),
        'meteor': Meteor()
    }